// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <stdio.h>

#include "rela/batch_runner.h"
#include "rela/prioritized_replay.h"
#include "rela/r2d2.h"

#include "rlcc/hanabi_env.h"
#include "rlcc/actors/actor.h"

class R2D2Actor: public Actor {
public:
    R2D2Actor(
            std::shared_ptr<rela::BatchRunner> runner,
            int seed,
            int numPlayer,                       // total number os players
            int playerIdx,                       // player idx for this player
            const std::vector<float>& epsList,   // list of eps to sample from
            const std::vector<float>& tempList,  // list of temp to sample from
            bool vdn,
            bool sad,
            bool shuffleColor,
            bool hideAction,
            bool trinary,  // trinary aux task or full aux
            std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer,
            // if replay buffer is None, then all params below are not used
            int multiStep,
            int seqLen,
            float gamma,
            std::vector<std::vector<std::vector<std::string>>> convention,
            bool actParameterized,
            int conventionIdx,
            int conventionOverride,
            bool conventionFictitiousOverride,
            bool useExperience,
            bool beliefStats)
        : Actor(
                seed,
                playerIdx,
                convention,
                conventionIdx,
                conventionOverride,
                false)
          , runner_(std::move(runner))
          , numPlayer_(numPlayer)
          , epsList_(epsList)
          , tempList_(tempList)
          , vdn_(vdn)
          , sad_(sad)
          , shuffleColor_(shuffleColor)
          , hideAction_(hideAction)
          , trinary_(trinary)
          , batchsize_(vdn_ ? numPlayer_ : 1)
          , playerEps_(batchsize_)
          , playerTemp_(batchsize_)
          , colorPermutes_(batchsize_)
          , invColorPermutes_(batchsize_)
          , replayBuffer_(std::move(replayBuffer))
          , r2d2Buffer_(std::make_unique<rela::R2D2Buffer>(multiStep, seqLen, gamma))
          , actParameterized_(actParameterized)
          , conventionFictitiousOverride_(conventionFictitiousOverride)
          , useExperience_(useExperience)
          , logStats_(false) 
          , beliefStats_(beliefStats) {
    }

    // simpler constructor for eval mode
    R2D2Actor(
            std::shared_ptr<rela::BatchRunner> runner,
            int numPlayer,
            int playerIdx,
            bool vdn,
            bool sad,
            bool hideAction,
            std::vector<std::vector<std::vector<std::string>>> convention,
            bool actParameterized,
            int conventionIdx,
            int conventionOverride,
            bool beliefStats)
        : Actor(
                1, // seed not used in eval move
                playerIdx,
                convention,
                conventionIdx,
                conventionOverride,
                true)
          , runner_(std::move(runner))
          , numPlayer_(numPlayer)
          , epsList_({0})
          , vdn_(vdn)
          , sad_(sad)
          , shuffleColor_(false)
          , hideAction_(hideAction)
          , trinary_(true)
          , batchsize_(vdn_ ? numPlayer_ : 1)
          , playerEps_(batchsize_)
          , colorPermutes_(batchsize_)
          , invColorPermutes_(batchsize_)
          , replayBuffer_(nullptr)
          , r2d2Buffer_(nullptr)
          , actParameterized_(actParameterized)
          , conventionFictitiousOverride_(false)
          , useExperience_(false)
          , logStats_(true) 
          , beliefStats_(beliefStats) {
    }

    virtual void addHid(rela::TensorDict& to, rela::TensorDict& hid);
    void reset(const HanabiEnv& env) override;
    void observeBeforeAct(HanabiEnv& env) override;
    void act(HanabiEnv& env, const int curPlayer) override;
    void fictAct(const HanabiEnv& env) override;
    void observeAfterAct(const HanabiEnv& env) override;

    void setPartners(std::vector<std::shared_ptr<R2D2Actor>> partners) {
        partners_ = std::move(partners);
        assert((int)partners_.size() == numPlayer_);
        assert(partners_[playerIdx_] == nullptr);
    }

    void setBeliefRunner(std::shared_ptr<rela::BatchRunner>& beliefModel) {
        assert(!vdn_ && batchsize_ == 1);
        beliefRunner_ = beliefModel;
        offBelief_ = true;
        // OBL does not need Other-Play, and does not support Other-Play
        assert(!shuffleColor_);
    }

    void setBeliefRunnerStats(std::shared_ptr<rela::BatchRunner>& beliefModel) {
        assert(!vdn_ && batchsize_ == 1);
        beliefRunner_ = beliefModel;
        // OBL does not need Other-Play, and does not support Other-Play
        assert(!shuffleColor_);
    }

    float getSuccessFictRate() {
        float rate = (float)successFict_ / totalFict_;
        successFict_ = 0;
        totalFict_ = 0;
        return rate;
    }

    void pushToReplayBuffer() override;

protected:
    rela::TensorDict getH0(int numPlayer, std::shared_ptr<rela::BatchRunner>& runner) {
        std::vector<torch::jit::IValue> input{numPlayer};
        auto model = runner->jitModel();
        auto output = model.get_method("get_h0")(input);
        auto h0 = rela::tensor_dict::fromIValue(output, torch::kCPU, true);
        return h0;
    }

    std::shared_ptr<rela::BatchRunner> runner_;
    std::shared_ptr<rela::BatchRunner> classifier_;
    const int numPlayer_;
    const std::vector<float> epsList_;
    const std::vector<float> tempList_;
    const bool vdn_;
    const bool sad_;
    const bool shuffleColor_;
    const bool hideAction_;
    const bool trinary_;
    const int batchsize_;

    std::vector<float> playerEps_;
    std::vector<float> playerTemp_;
    std::vector<std::vector<int>> colorPermutes_;
    std::vector<std::vector<int>> invColorPermutes_;

    std::shared_ptr<rela::RNNPrioritizedReplay> replayBuffer_;
    std::unique_ptr<rela::R2D2Buffer> r2d2Buffer_;

    rela::TensorDict prevHidden_;
    rela::TensorDict hidden_;

    rela::FutureReply futReply_;
    rela::FutureReply futPriority_;
    rela::FutureReply fictReply_;
    rela::FutureReply futReward_;
    rela::RNNTransition lastEpisode_;

    bool offBelief_ = false;
    std::shared_ptr<rela::BatchRunner> beliefRunner_;
    rela::TensorDict beliefHidden_;
    rela::FutureReply futBelief_;

    std::vector<int> privCardCount_;
    std::vector<hle::HanabiCardValue> sampledCards_;
    rela::FutureReply futTarget_;

    int totalFict_ = 0;
    int successFict_ = 0;
    bool validFict_ = false;
    std::unique_ptr<hle::HanabiState> fictState_ = nullptr;
    std::vector<std::shared_ptr<R2D2Actor>> partners_;

    bool actParameterized_;
    bool conventionFictitiousOverride_;
    bool useExperience_;
    bool logStats_;
    bool beliefStats_;
};
