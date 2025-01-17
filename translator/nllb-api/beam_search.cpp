#include "translator/nllb-api/beam_search.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <spdlog/spdlog.h>

namespace nllb {

BeamSearchDecoder::BeamSearchDecoder(int beam_size, float length_penalty, float eos_token_id)
    : beam_size_(beam_size)
    , length_penalty_(length_penalty)
    , eos_token_id_(eos_token_id) {}

} // namespace nllb 