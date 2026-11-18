#ifndef RHS_ACK_H
#define RHS_ACK_H

#include <stdint.h>

#include "frame.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RHS_ACK_MODEL_MAX 32u

typedef enum {
    RHS_ACK_IDLE = 0,
    RHS_ACK_AWAIT_RECV_OK = 1,
    RHS_ACK_AWAIT_DATA_ACK = 2,
    RHS_ACK_DELIVERED = 3,
    RHS_ACK_FAILED = 4
} rhs_ack_state_t;

typedef struct {
    uint32_t seq;
    rhs_ack_state_t state;
    uint8_t attempts;
    uint8_t max_attempts;
    char model[RHS_ACK_MODEL_MAX];
} rhs_ack_t;

void rhs_ack_init(rhs_ack_t *ack, uint8_t max_attempts);
void rhs_ack_data_request(rhs_ack_t *ack, uint32_t seq);
int rhs_ack_on_frame(rhs_ack_t *ack, const rhs_frame_t *frame);
int rhs_ack_retry(rhs_ack_t *ack);
int rhs_ack_complete(const rhs_ack_t *ack);

#ifdef __cplusplus
}
#endif

#endif /* RHS_ACK_H */
