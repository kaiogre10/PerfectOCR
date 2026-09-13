#pragma once
#include <vector>
#include <cstdint>
#include <queue>

namespace payload {
	void container_create();
	void push(std::vector<uint8_t>&& plain_payload);
	std::queue<std::vector<uint8_t>> drain();
}

