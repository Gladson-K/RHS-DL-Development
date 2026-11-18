#include "crc32.h"

uint32_t rhs_crc32(const uint8_t *data, size_t length) {
    uint32_t crc = 0xFFFFFFFFu;
    size_t i;

    if (data == 0) {
        return 0u;
    }
    for (i = 0; i < length; ++i) {
        int bit;
        crc ^= (uint32_t)data[i];
        for (bit = 0; bit < 8; ++bit) {
            uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}
