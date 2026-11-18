#include "ack.h"

#include <string.h>

void rhs_ack_init(rhs_ack_t *ack, uint8_t max_attempts) {
    if (ack == 0) {
        return;
    }
    ack->seq = 0u;
    ack->state = RHS_ACK_IDLE;
    ack->attempts = 0u;
    ack->max_attempts = (max_attempts == 0u) ? 3u : max_attempts;
    ack->model[0] = '\0';
}

void rhs_ack_data_request(rhs_ack_t *ack, uint32_t seq) {
    if (ack == 0) {
        return;
    }
    ack->seq = seq;
    ack->state = RHS_ACK_AWAIT_RECV_OK;
    ack->attempts = 0u;
    ack->model[0] = '\0';
}

int rhs_ack_on_frame(rhs_ack_t *ack, const rhs_frame_t *frame) {
    if (ack == 0 || frame == 0) {
        return -1;
    }
    if (frame->type == (uint8_t)RHS_ERROR) {
        ack->state = RHS_ACK_FAILED;
        return -1;
    }
    if (ack->seq != frame->seq) {
        return 0;
    }
    if (frame->type == (uint8_t)RHS_RECV_OK && ack->state == RHS_ACK_AWAIT_RECV_OK) {
        ack->state = RHS_ACK_AWAIT_DATA_ACK;
    } else if (frame->type == (uint8_t)RHS_DATA_ACK && ack->state == RHS_ACK_AWAIT_DATA_ACK) {
        ack->state = RHS_ACK_DELIVERED;
    }
    return 0;
}

int rhs_ack_retry(rhs_ack_t *ack) {
    if (ack == 0) {
        return 0;
    }
    if (ack->state != RHS_ACK_AWAIT_RECV_OK && ack->state != RHS_ACK_AWAIT_DATA_ACK) {
        return 0;
    }
    if ((uint8_t)(ack->attempts + 1u) >= ack->max_attempts) {
        ack->attempts = ack->max_attempts;
        ack->state = RHS_ACK_FAILED;
        return 0;
    }
    ack->attempts = (uint8_t)(ack->attempts + 1u);
    return 1;
}

int rhs_ack_complete(const rhs_ack_t *ack) {
    return ack != 0 && ack->state == RHS_ACK_DELIVERED;
}
