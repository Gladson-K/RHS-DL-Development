#include "ack.h"
#include "crc32.h"
#include "frame.h"
#include "jev_probe.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void test_crc_reference(void) {
    /* CRC32("123456789") == 0xCBF43926 (IEEE 802.3 reference vector). */
    const uint8_t data[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
    assert(rhs_crc32(data, sizeof(data)) == 0xCBF43926u);
}

static void test_frame_roundtrip(void) {
    const uint8_t payload[] = "{\"sample_id\":\"s00001\"}";
    uint8_t buffer[128];
    rhs_frame_t frame;
    size_t consumed = 0;
    size_t written;

    written = rhs_frame_encode(RHS_DATA_REQ, 42u, payload, (uint32_t)(sizeof(payload) - 1u),
                               buffer, sizeof(buffer));
    assert(written == RHS_HEADER_SIZE + sizeof(payload) - 1u + RHS_CRC_SIZE);
    assert(rhs_frame_decode(buffer, written, &frame, &consumed) == RHS_FRAME_OK);
    assert(consumed == written);
    assert(frame.type == RHS_DATA_REQ);
    assert(frame.seq == 42u);
    assert(frame.length == sizeof(payload) - 1u);
    assert(memcmp(frame.payload, payload, frame.length) == 0);

    assert(rhs_frame_decode(buffer, written - 1u, &frame, &consumed) == RHS_FRAME_INCOMPLETE);
    buffer[8] ^= 0xFFu;
    assert(rhs_frame_decode(buffer, written, &frame, &consumed) == RHS_FRAME_CORRUPT);
}

static void test_ack_machine(void) {
    rhs_ack_t ack;
    rhs_frame_t frame;

    rhs_ack_init(&ack, 3u);
    rhs_ack_data_request(&ack, 7u);
    assert(ack.state == RHS_ACK_AWAIT_RECV_OK);

    memset(&frame, 0, sizeof(frame));
    frame.type = RHS_RECV_OK;
    frame.seq = 7u;
    rhs_ack_on_frame(&ack, &frame);
    assert(ack.state == RHS_ACK_AWAIT_DATA_ACK);

    frame.type = RHS_DATA_ACK;
    rhs_ack_on_frame(&ack, &frame);
    assert(ack.state == RHS_ACK_DELIVERED);
    assert(rhs_ack_complete(&ack) == 1);
}

static void test_jev_signature(void) {
    float gram[RHS_JEV_DIM][RHS_JEV_DIM];
    float signature[RHS_JEV_K];
    float total = 0.0f;
    int i;
    int j;

    for (i = 0; i < RHS_JEV_DIM; ++i) {
        for (j = 0; j < RHS_JEV_DIM; ++j) {
            gram[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    rhs_jev_signature(gram, signature);
    for (i = 0; i < RHS_JEV_K; ++i) {
        assert(signature[i] >= 0.0f);
        total += signature[i];
    }
    assert(fabsf(total - 1.0f) < 1e-5f);
    for (i = 0; i < RHS_JEV_K; ++i) {
        assert(fabsf(signature[i] - 0.25f) < 1e-4f);
    }
}

int main(void) {
    test_crc_reference();
    test_frame_roundtrip();
    test_ack_machine();
    test_jev_signature();
    printf("firmware protocol tests: OK\n");
    return 0;
}
