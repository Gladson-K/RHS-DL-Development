#ifndef RHS_CRC32_H
#define RHS_CRC32_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320) — identical to zlib. */
uint32_t rhs_crc32(const uint8_t *data, size_t length);

#ifdef __cplusplus
}
#endif

#endif /* RHS_CRC32_H */
