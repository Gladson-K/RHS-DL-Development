#ifndef RHS_FRAME_H
#define RHS_FRAME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RHS_MAGIC_0 'R'
#define RHS_MAGIC_1 'H'
#define RHS_MAGIC_2 'S'
#define RHS_MAGIC_3 '1'
#define RHS_VERSION 1u
#define RHS_HEADER_SIZE 14u /* magic(4) + version(1) + type(1) + seq(4) + len(4) */
#define RHS_CRC_SIZE 4u
#define RHS_MAX_PAYLOAD (1024u * 1024u)

typedef enum {
    RHS_DATA_REQ = 1,
    RHS_RECV_OK = 2,
    RHS_DATA_ACK = 3,
    RHS_JEV_DECISION = 4,
    RHS_MODEL_SWITCH = 5,
    RHS_RESULT = 6,
    RHS_ERROR = 7
} rhs_frame_type_t;

typedef enum {
    RHS_FRAME_OK = 0,
    RHS_FRAME_INCOMPLETE = 1,
    RHS_FRAME_CORRUPT = -1
} rhs_frame_status_t;

typedef struct {
    uint8_t version;
    uint8_t type;
    uint32_t seq;
    uint32_t length;
    const uint8_t *payload;
} rhs_frame_t;

/* Encode one frame. Returns the number of bytes written, or 0 on error. */
size_t rhs_frame_encode(rhs_frame_type_t type,
                        uint32_t seq,
                        const uint8_t *payload,
                        uint32_t length,
                        uint8_t *out,
                        size_t out_capacity);

/* Decode one frame. Returns RHS_FRAME_OK/INCOMPLETE/CORRUPT. */
int rhs_frame_decode(const uint8_t *buffer,
                     size_t length,
                     rhs_frame_t *out,
                     size_t *consumed);

#ifdef __cplusplus
}
#endif

#endif /* RHS_FRAME_H */
