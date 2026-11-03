"""Normalize only CDR alignment padding for PointCloud2 wire comparisons.

Fast-CDR does not promise initialized padding bytes, even when serializing the
same message twice. Preserve every meaningful field and point payload byte.
"""
import struct


def canonical_pointcloud_wire(serialized):
    raw = bytearray(serialized)
    assert raw[:2] == b'\x00\x01', 'Expected little-endian CDR encapsulation'
    offset = 4
    padding = []
    def align(n):
        nonlocal offset
        end = offset+(-(offset-4) % n)
        assert end <= len(raw)
        padding.extend(range(offset, end)); raw[offset:end] = b'\x00'*(end-offset); offset = end
    def uint32():
        nonlocal offset
        align(4); value = struct.unpack_from('<I', raw, offset)[0]; offset += 4
        return value
    def string():
        nonlocal offset
        count = uint32(); assert count > 0 and offset+count <= len(raw)
        assert raw[offset+count-1] == 0
        offset += count
    uint32(); uint32(); string()  # header stamp and frame_id
    uint32(); uint32()  # height, width
    count = uint32()
    assert count <= 100
    for _ in range(count):
        string(); uint32(); offset += 1; uint32()  # PointField
    offset += 1  # is_bigendian
    uint32(); uint32()  # point_step, row_step
    count = uint32(); offset += count  # data sequence
    assert offset < len(raw) and raw[offset] in (0, 1)
    offset += 1  # is_dense
    assert offset == len(raw), (offset, len(raw))
    return bytes(raw), padding
