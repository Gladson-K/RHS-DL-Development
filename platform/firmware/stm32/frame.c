#include "frame.h"
#include "crc32.h"

#include <string.h>

static void put_be32(uint8_t *out, uint32_t value) {
    out[0] = (uint8_t)((value >> 24) & 0xFFu);
    out[1] = (uint8_t)((value >> 16) & 0xFFu);
    out[2] = (uint8_t)((value >> 8) & 0xFFu);
    out[3] = (uint8_t)(value & 0xFFu);
}

static uint32_t get_be32(const uint8_t *in) {
    return ((uint32_t)in[0] << 24) | ((uint32_t)in[1] << 16) |
           ((uint32_t)in[2] << 8) | (uint32_t)in[3];
}

size_t rhs_frame_encode(rhs_frame_type_t type,
                        uint32_t seq,
                        const uint8_t *payload,
                        uint32_t length,
                        uint8_t *out,
                        size_t out_capacity) {
    size_t total;
    uint32_t crc;

    if (out == 0 || length > RHS_MAX_PAYLOAD) {
        return 0u;
    }
    if (payload == 0 && length > 0u) {
        return 0u;
    }
    total = (size_t)RHS_HEADER_SIZE + (size_t)length + (size_t)RHS_CRC_SIZE;
    if (out_capacity < total) {
        return 0u;
    }
    out[0] = RHS_MAGIC_0;
    out[1] = RHS_MAGIC_1;
    out[2] = RHS_MAGIC_2;
    out[3] = RHS_MAGIC_3;
    out[4] = (uint8_t)RHS_VERSION;
    out[5] = (uint8_t)type;
    put_be32(out + 6, seq);
    put_be32(out + 10, length);
    if (length > 0u) {
        memcpy(out + RHS_HEADER_SIZE, payload, length);
    }
    crc = rhs_crc32(out, (size_t)RHS_HEADER_SIZE + (size_t)length);
    put_be32(out + RHS_HEADER_SIZE + length, crc);
    return total;
}

int rhs_frame_decode(const uint8_t *buffer,
                     size_t length,
                     rhs_frame_t *out,
                     size_t *consumed) {
    uint8_t version;
    uint8_t type;
    uint32_t seq;
    uint32_t payload_length;
    uint32_t expected;
    uint32_t actual;
    size_t total;

    if (buffer == 0 || out == 0) {
        return RHS_FRAME_CORRUPT;
    }
    if (length < RHS_HEADER_SIZE) {
        return RHS_FRAME_INCOMPLETE;
    }
    if (buffer[0] != RHS_MAGIC_0 || buffer[1] != RHS_MAGIC_1 ||
        buffer[2] != RHS_MAGIC_2 || buffer[3] != RHS_MAGIC_3) {
        return RHS_FRAME_CORRUPT;
    }
    version = buffer[4];
    type = buffer[5];
    seq = get_be32(buffer + 6);
    payload_length = get_be32(buffer + 10);
    if (version != (uint8_t)RHS_VERSION || payload_length > RHS_MAX_PAYLOAD) {
        return RHS_FRAME_CORRUPT;
    }
    total = (size_t)RHS_HEADER_SIZE + (size_t)payload_length + (size_t)RHS_CRC_SIZE;
    if (length < total) {
        return RHS_FRAME_INCOMPLETE;
    }
    expected = get_be32(buffer + RHS_HEADER_SIZE + payload_length);
    actual = rhs_crc32(buffer, (size_t)RHS_HEADER_SIZE + (size_t)payload_length);
    if (expected != actual) {
        return RHS_FRAME_CORRUPT;
    }
    out->version = version;
    out->type = type;
    out->seq = seq;
    out->length = payload_length;
    out->payload = buffer + RHS_HEADER_SIZE;
    if (consumed != 0) {
        *consumed = total;
    }
    return RHS_FRAME_OK;
}
