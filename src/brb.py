import os
import struct
import logging
import mmap

class BRB:
    """Block Ring Buffer
    Given a directory, a maximum size in bytes and a number of blocks, BRB
    maintains a circular record-based buffer of arbitrary data.
    """
    _magicnumber_block = 0xa558133d
    _magicnumber_record = 0xbeefcafe
    _version = 1

    def __init__(self, basedir, maxsize, blockcount):
        self._basedir = basedir
        self._maxsize = maxsize
        self._blockcount = blockcount
        self._blocksize = maxsize / blockcount

        self._currentblock = None

    def _scanblocks(self):
        """Examines all the blocks and returns a tuple of dicts identifying the
        blocks with the lowest and highest seequence numbers, which correspond
        to the oldest and newest blocks. If a sequence number is -1, the block
        does not exist. In any case, writing to the low block is recommended."""

        low = None
        high = None

        for blocknum in xrange(self._blockcount):
            try:
                sequence = self._identify(blocknum)
            except IOError, ex:
                sequence = -1

            if low is None or sequence < low['sequence']:
                low = {'blocknum': blocknum, 'sequence': sequence}
            
            if high is None or sequence > high['sequence']:
                high = {'blocknum': blocknum, 'sequence': sequence}

        return low, high

    def _identify(self, blocknum):
        """"Read the sequence number from a block identified by `blocknum`"""
        header = struct.Struct(">LLQ")
        with open(self._blockpath(blocknum)) as fh:
            (magicnumber, version, sequence) = header.unpack(fh.read(header.size))
        assert magicnumber == self._magicnumber_block
        assert version == self._version
        return sequence

    def _blockheader(self, sequence):
        """Return a complete header for a new block with sequence number `sequence`."""
        return struct.pack(">LLQ", self._magicnumber_block, self._version, sequence)
    
    def _blockpath(self, blocknum):
        """Return the path to where a block with `blocknum` can be found."""
        return os.path.join(self._basedir, "0x%08x.block" % blocknum)
  
class BRBReader(BRB):
    def __iter__(self):
        first, last = self._scanblocks()
        for (blocknum, start, timestamp, data) in self._iter(first['blocknum']):
            yield timestamp, data

    def _iter(self, block, offset = 0, direction = 1):
        """A generator that yields the byte data for each record, oldest to newest."""
        blocknum = block

        while True:
            blockpath = self._blockpath(blocknum)

            logging.debug("Opening block %d from %s" % (blocknum, blockpath))
            fh = open(blockpath)

            # mmap the entire block
            mapped = mmap.mmap(fh.fileno(), 0, access = mmap.ACCESS_READ)
    
            # Give the user slices of the block data between record offsets.
            if direction == 1:
                offsets = self._getOffsets(mapped, struct.pack(">L", self._magicnumber_record), start = offset)
            else:
                offsets = self._rGetOffsets(mapped, struct.pack(">L", self._magicnumber_record))
                
            for (start, end) in offsets:
                logging.debug("Found record: %s:%d-%d" % (blockpath, start, end))
                timestamp = struct.unpack(">d", mapped[start:start + struct.calcsize(">d")])[0]
                yield blocknum, start, timestamp, mapped[start + struct.calcsize(">d"):end]

            del mapped
            fh.close()

            # Advance to the next block, but stop when we wrap around to the starting block.
            blocknum = (blocknum + direction) % self._blockcount
            print "block change, %d" % direction
            if blocknum == -1:
                blocknum = self._blockcount
            if blocknum == block:
                raise StopIteration

    def read(self, timestamp_wanted):
        first, last = self._scanblocks()

        f = self._iter(first['blocknum']).next()
        l = self._iter(last['blocknum'], direction = -1).next()
        
        if timestamp_wanted < f[2]:
            return f[2], f[3]

        if timestamp_wanted > l[2]:
            return l[2], l[3]

        unitvalue = (timestamp_wanted - f[2]) / (l[2] - f[2])  
        #print "unit value", unitvalue

        globaloffset = int(unitvalue * self._maxsize)
        #print "want global offset", globaloffset

        #print "global block offset", globaloffset / self._blocksize

        block = (f[0] + (globaloffset / self._blocksize)) % self._blockcount
        #print "mapped block", block

        offset = globaloffset % self._blocksize
        #print "mapped offset", offset

        checked = 0
        i = self._iter(block, offset)

        (b, o, t, d) = i.next()
        checked += 1
        if t < timestamp_wanted:
            # iter forwards
            last = None
            for (b, o, t, d) in self._iter(block, offset):
                checked += 1
                if t > timestamp_wanted:
                    print last[0], "%d records checked, %d -> %d" % (checked, block, b)
                    return last
                last = (t, d)
        elif t > timestamp_wanted:
            # iter backwards
            last = None
            for (b, o, t, d) in self._iter(block, offset, -1):
                checked += 1
                if t < timestamp_wanted:
                    print last[0], "%d records checked backwards, %d -> %d" % (checked, block, b)
                    return last
                last = (t, d)
        else:
            print "Exact match?"
            return (t, d)
            

        #return self.read_(timestamp_wanted)

    def read_(self, timestamp_wanted):
        """Naive linear search"""
        before = None
        after = None
        last = None

        for (block, offset, timestamp, data) in self.iter():
            last = (timestamp, data)
            if before is None:
                before = (timestamp, data)
                continue

            if after is None:
                after = (timestamp, data)

            if timestamp_wanted < before[0]:
                print "before", timestamp_wanted, before[0]
                return before
            elif before[0] < timestamp_wanted < after[0]:
                print "mid"
                return before
                # TODO calculate distance and decide on that
            else:
                print "move"
                before = after
                after = None

        return last

    def _getOffsets(self, mapped, magicbytes, start = 0):
        """A generator that yields pairs of byte offsets in `mapped` corresponding to the first and last bytes between appearances of `magicbytes`. For example, 'xABCxDEF' would produce [(1, 4), (5, 8)]"""
        start = mapped.find(magicbytes, start)
        while True:
            end = mapped.find(magicbytes, start + len(magicbytes))
            if end == -1:
                yield (start + len(magicbytes), mapped.size())
                raise StopIteration

            yield (start + len(magicbytes), end)
            start = end

    def _rGetOffsets(self, mapped, magicbytes):
        """Like _getOffsets, except backwards."""
        end = mapped.size()
        while True:
            start = mapped.rfind(magicbytes, 0, end)
            if start == -1:
                raise StopIteration
            yield start + len(magicbytes), end
            end = start


class BRBWriter(BRB):
    def _nextblock(self):
        """Examines all the blocks, identifies the next one to create or truncate,
        and opens the file. Returns a dict identifying that block."""

        low, high = self._scanblocks()

        # The next block to be written to is the one with the lowest
        # sequence number. Write to the block number that contains it,
        # and assign it the sequence number after the highest one seen.
        # Blocks that don't exist are considered to have a sequence number
        # of -1, so they will always be first.
        block = {'blocknum': low['blocknum'], 'sequence': high['sequence'] + 1}

        # Open/create/truncate the block and write the new header.
        block['fh'] = open(self._blockpath(block['blocknum']), "w+")
        block['fh'].write(self._blockheader(sequence = block['sequence']))

        logging.debug("New block at %s: sequence %d" % (self._blockpath(block['blocknum']), block['sequence']))

        return block
       
    def write(self, timestamp, data):
        """Write a string of bytes, creating, overwriting and advancing to new blocks as necessary."""
        if self._currentblock is None:
            self._currentblock = self._nextblock()
        elif self._currentblock['fh'].tell() + len(data) > self._blocksize:
            # If writing this record would exceed the blocksize, close this block
            # and move to the next one.
            logging.debug("Ending block %s because record (%d bytes) would exceed blocksize (%d > %d)",
                self._blockpath(self._currentblock['blocknum']),
                len(data),
                self._currentblock['fh'].tell() + len(data),
                self._blocksize)

            self._currentblock['fh'].close()
            self._currentblock = self._nextblock()

        logging.debug("Writing %d bytes to %s", len(data), self._blockpath(self._currentblock['blocknum']))
        self._currentblock['fh'].write(struct.pack(">Ld", self._magicnumber_record, timestamp) + data)

if __name__ == '__main__':
    import sys
    logging.getLogger().setLevel(logging.DEBUG)
    buf = BRBWriter(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    for line in sys.stdin.readlines():
        buf.write(line)


