import os
import struct
import logging

class BRBWriter:
    """Block Ring Buffer
    Given a directory, a maximum size in bytes and a number of blocks, BRB
    maintains a circular buffer of arbitrary data.
    """
    _magicnumber = 0xa558133d
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

    def _identify(self, blocknum):
        """"Read the sequence number from a block identified by `blocknum`"""
        header = struct.Struct(">LLQ")
        with open(self._blockpath(blocknum)) as fh:
            (magicnumber, version, sequence) = header.unpack(fh.read(header.size))
        assert magicnumber == self._magicnumber
        assert version == self._version
        return sequence

    def _blockheader(self, sequence):
        """Return a complete header for a new block with sequence number `sequence`."""
        return struct.pack(">LLQ", self._magicnumber, self._version, sequence)
    
    def _blockpath(self, blocknum):
        """Return the path to where a block with `blocknum` can be found."""
        return os.path.join(self._basedir, "0x%08x.block" % blocknum)
        
    def write(self, bytes):
        """Write a string of bytes, creating, overwriting and advancing to new blocks as necessary."""
        if self._currentblock is None:
            self._currentblock = self._nextblock()
        elif self._currentblock['fh'].tell() + len(bytes) > self._blocksize:
            # If writing this record would exceed the blocksize, close this block
            # and move to the next one.
            logging.debug("Ending block %s because record (%d bytes) would exceed blocksize (%d > %d)",
                self._blockpath(self._currentblock['blocknum']),
                len(bytes),
                self._currentblock['fh'].tell() + len(bytes),
                self._blocksize)

            self._currentblock['fh'].close()
            self._currentblock = self._nextblock()

        logging.debug("Writing %d bytes to %s", len(bytes), self._blockpath(self._currentblock['blocknum']))
        self._currentblock['fh'].write(bytes)

if __name__ == '__main__':
    import sys
    logging.getLogger().setLevel(logging.DEBUG)
    buf = BRBWriter(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    for line in sys.stdin.readlines():
        buf.write(line)


