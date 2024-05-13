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

class R2D2Actor {
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
        int multiStep,
        int seqLen,
        float gamma,
        std::vector<std::vector<std::string>> conventions,
        bool actParameterized,
        int conventionIdx,
        int conventionOverride,
        bool conventionFictitiousOverride,
        bool useExperience,
        bool beliefStats,
        bool sadLegacy,
        bool iqlLegacy,
        bool beliefSadLegacy,
        bool colorShuffleSync,
        int partnerIdx,
        int numPartners,
        std::unordered_map<std::string,int> colourPermutationMap,
        std::vector<std::vector<int>> allColourPermutations,
        std::vector<std::vector<int>> allInvColourPermutations,
        bool distShuffleColour,
        std::vector<std::vector<float>> permutationDistribution,
        bool shuffleConvention)
          : runner_(std::move(runner))
            , rng_(seed)
            , numPlayer_(numPlayer)
            , playerIdx_(playerIdx)
            , epsList_(epsList)
            , tempList_(tempList)
            , vdn_(vdn)
            , sad_(sad)
            , shuffleColor_(shuffleColor)
            , hideAction_(hideAction)
            , trinary_(trinary)
            , batchsize_(vdn_ ? numPlayer : 1)
            , playerEps_(batchsize_)
            , playerTemp_(batchsize_)
            , colorPermutes_(batchsize_)
            , invColorPermutes_(batchsize_)
            , replayBuffer_(std::move(replayBuffer))
            , r2d2Buffer_(std::make_unique<rela::R2D2Buffer>(multiStep, seqLen, gamma))

            // My changes
            , conventions_(conventions) 
            , conventionIdx_(conventionIdx) 
            , conventionOverride_(conventionOverride) 
            , conventionFictitiousOverride_(conventionFictitiousOverride)
            , actParameterized_(actParameterized)
            , recordStats_(false)
            , useExperience_(useExperience)
            , logStats_(false) 
            , livesBeforeMove_(-1) 
            , currentTwoStep_("X") 
            , beliefStats_(beliefStats) 
            , sadLegacy_(sadLegacy) 
            , iqlLegacy_(iqlLegacy) 
            , beliefSadLegacy_(beliefSadLegacy) 
            , sentSignal_(false)
            , sentSignalStats_(false)
            , beliefStatsSignalReceived_(false) 
            , colorShuffleSync_(colorShuffleSync) 
            , colourPermuteConstant_(false) 
            , partnerIdx_(partnerIdx) 
            , numPartners_(numPartners) 
            , colourPermutationMap_(colourPermutationMap) 
            , allColourPermutations_(allColourPermutations)
            , allInvColourPermutations_(allInvColourPermutations)
            , distShuffleColour_(distShuffleColour) 
            , permutationDistribution_(permutationDistribution) 
            , shuffleConvention_(shuffleConvention) {
      if (beliefStats_ && conventions_.size() > 0) {
        auto responseMove = strToMove(conventions_[conventionIdx_][1]);
        beliefStatsResponsePosition_ = responseMove.CardIndex();
      }

      if (sadLegacy_) {
        showOwnCards_ = false;
      } else {
        showOwnCards_ = true;
      }
    }


    // simpler constructor for eval mode
    R2D2Actor(
        std::shared_ptr<rela::BatchRunner> runner,
        int numPlayer,
        int playerIdx,
        bool vdn,
        bool sad,
        bool hideAction,
        std::vector<std::vector<std::string>> conventions,
        bool actParameterized,
        int conventionIdx,
        int conventionOverride,
        bool beliefStats,
        bool sadLegacy,
        bool iqlLegacy,
        bool shuffleColor,
        std::vector<std::vector<int>> allColourPermutations,
        std::vector<std::vector<int>> allInvColourPermutations,
        bool distShuffleColour,
        std::vector<std::vector<float>> permutationDistribution,
        int partnerIdx,
        int seed,
        bool shuffleConvention)
      : runner_(std::move(runner))
        , rng_(seed)
        , numPlayer_(numPlayer)
        , playerIdx_(playerIdx)
        , epsList_({0})
        , vdn_(vdn)
        , sad_(sad)
        , shuffleColor_(shuffleColor)
        , hideAction_(hideAction)
        , trinary_(true)
        , batchsize_(vdn_ ? numPlayer : 1)
        , playerEps_(batchsize_)
        , colorPermutes_(batchsize_)
        , invColorPermutes_(batchsize_)
        , replayBuffer_(nullptr)
        , r2d2Buffer_(nullptr)

        // My changes
        , conventions_(conventions) 
        , conventionIdx_(conventionIdx) 
        , conventionOverride_(conventionOverride) 
        , conventionFictitiousOverride_(false)
        , actParameterized_(actParameterized)
        , recordStats_(true)
        , useExperience_(false)
        , logStats_(true) 
        , livesBeforeMove_(-1) 
        , currentTwoStep_("X") 
        , beliefStats_(beliefStats) 
        , sadLegacy_(sadLegacy) 
        , iqlLegacy_(iqlLegacy) 
        , sentSignal_(false)
        , sentSignalStats_(false)
        , beliefStatsSignalReceived_(false)
        , colorShuffleSync_(false) 
        , colourPermuteConstant_(false) 
        , partnerIdx_(partnerIdx)
        , numPartners_(0) 
        , allColourPermutations_(allColourPermutations)
        , allInvColourPermutations_(allInvColourPermutations)
        , distShuffleColour_(distShuffleColour) 
        , permutationDistribution_(permutationDistribution) 
        , shuffleConvention_(shuffleConvention) {
        if (beliefStats_ && conventions_.size() > 0) {
            auto responseMove = strToMove(conventions_[conventionIdx_][1]);
            beliefStatsResponsePosition_ = responseMove.CardIndex();
        }

        if (sadLegacy_) {
            showOwnCards_ = false;
        } else {
            showOwnCards_ = true;
        }
    }
    
    void setPartners(std::vector<std::shared_ptr<R2D2Actor>> partners) {
      partners_.reserve(partners.size());
      for (size_t i = 0; i < partners.size(); i++) {
        if ((int)i == playerIdx_) {
          partners_.push_back(std::weak_ptr<R2D2Actor>());
          continue;
        }
        partners_.emplace_back(std::move(partners[i]));
      }
      assert((int)partners_.size() == numPlayer_);
    }

    void reset(const HanabiEnv& env);
    void observeBeforeAct(HanabiEnv& env);
    void act(HanabiEnv& env, const int curPlayer);
    void fictAct(const HanabiEnv& env);
    void observeAfterAct(const HanabiEnv& env);

    void addHid(rela::TensorDict& to, rela::TensorDict& hid);
    void moveHid(rela::TensorDict& from, rela::TensorDict& hid);

    void setBeliefRunner(std::shared_ptr<rela::BatchRunner>& beliefModel) {
      assert(!vdn_ && batchsize_ == 1);
      beliefRunner_ = beliefModel;
      offBelief_ = true;
      // OBL does not need Other-Play, and does not support Other-Play
      assert(!shuffleColor_);
    }

    float getSuccessFictRate() {
      float rate = (float)successFict_ / totalFict_;
      successFict_ = 0;
      totalFict_ = 0;
      return rate;
    }

    std::tuple<int, int, int, int> getPlayedCardInfo() const {
      return {noneKnown_, colorKnown_, rankKnown_, bothKnown_};
    }

    // My changes
    
    void setBeliefRunnerStats(std::shared_ptr<rela::BatchRunner>& beliefModel) {
      assert(!vdn_ && batchsize_ == 1);
      beliefRunner_ = beliefModel;
      // OBL does not need Other-Play, and does not support Other-Play
      assert(!shuffleColor_);
    }

    void pushToReplayBuffer();
    
    std::unordered_map<std::string, float> getStats() const { return stats_; }
    rela::TensorDict getGameStory() const { return gameStory_; };

    int getConventionIndex() { return conventionIdx_; }

    void setCompareRunners(
        std::vector<std::shared_ptr<rela::BatchRunner>> compRunners,
        std::vector<bool> compSad,
        std::vector<bool> compSadLegacy,
        std::vector<bool> compIqlLegacy,
        std::vector<bool> compHideAction,
        std::vector<std::string> compNames) {

      for (auto compRunner: compRunners) {
        compRunners_.push_back(std::move(compRunner));
        compHidden_.push_back(rela::TensorDict());
        compFutReply_.push_back(rela::FutureReply());
      }

      compSad_ = compSad;
      compSadLegacy_ = compSadLegacy;
      compIqlLegacy_ = compIqlLegacy;
      compHideAction_ = compHideAction;
      compNames_ = compNames;
    }

    void setColourPermute(
        std::vector<std::vector<int>> colorPermute,
        std::vector<std::vector<int>> invColorPermute,
        std::vector<int> shuffleIndex,
        std::vector<bool> compShuffleColor,
        std::vector<std::vector<int>> compColorPermute,
        std::vector<std::vector<int>> compInvColorPermute) {
      colourPermuteConstant_ = true;
      colorPermutes_ = colorPermute;
      invColorPermutes_ = invColorPermute;
      compShuffleColor_ = compShuffleColor;
      shuffleIndex_ = shuffleIndex;
      compColorPermutes_ = compColorPermute;
      compInvColorPermutes_ = compInvColorPermute;
    }

    void setPermutationDistribution(
        std::vector<std::vector<float>> permutationDistribution) {
      permutationDistribution_ = permutationDistribution;
    }

  private:
    rela::TensorDict getH0(int numPlayer, std::shared_ptr<rela::BatchRunner>& runner) {
      std::vector<torch::jit::IValue> input{numPlayer};
      auto model = runner->jitModel();
      auto output = model.get_method("get_h0")(input);
      auto h0 = rela::tensor_dict::fromIValue(output, torch::kCPU, true);
      return h0;
    }

    // My changes
    void conventionReset(const HanabiEnv& env);
    void incrementStat(std::string key);
    virtual void incrementStatsBeforeMove(
        const HanabiEnv& env, hle::HanabiMove move);
    void incrementStatsConvention(const HanabiEnv& env, hle::HanabiMove move);
    void incrementStatsConventionRole(bool shouldHavePlayed, std::string role,
        hle::HanabiMove movePlayed, hle::HanabiMove moveRole);
    void incrementStatsTwoStep(const HanabiEnv& env, hle::HanabiMove move);
    std::string conventionString();
    virtual void incrementStatsAfterMove(const HanabiEnv& env);
    hle::HanabiMove overrideMove(const HanabiEnv& env, hle::HanabiMove move,
        std::vector<float> actionQ, bool exploreAction,
        std::vector<float> legalMoves);
    bool moveInVector(std::vector<hle::HanabiMove> moves, hle::HanabiMove move);
    std::tuple<hle::HanabiMove, bool> availableSignalMove( const HanabiEnv& env, 
        std::vector<std::vector<std::string>> conventions);
    hle::HanabiMove matchingResponseMove(
        std::vector<std::vector<std::string>> convention,
        hle::HanabiMove signalMove);
    hle::HanabiMove different_action(const HanabiEnv& env, 
        std::vector<hle::HanabiMove> exclude,
        std::vector<float> actionQ, bool exploreAction,
        std::vector<float> legalMoves);
    bool movePlayableOnFireworks(const HanabiEnv& env, hle::HanabiMove move, int player);
    bool discardMovePlayable(const HanabiEnv& env, hle::HanabiMove move);
    hle::HanabiMove strToMove(std::string key);
    std::tuple<int,int,std::vector<int>> beliefConventionPlayable(
        const HanabiEnv& env);
    bool playedCardPossiblySignalledCard(hle::HanabiMove playedMove,
        std::shared_ptr<hle::HanabiHand> playedHand);
    void possibleResponseCards(const HanabiEnv& env,
        std::vector<int>& playableCards);
    void callCompareAct(HanabiEnv& env);
    void replyCompareAct(const HanabiEnv& env, int actorAction, int curPlayer);

    std::shared_ptr<rela::BatchRunner> runner_;
    std::shared_ptr<rela::BatchRunner> classifier_;
    std::mt19937 rng_;
    const int numPlayer_;
    const int playerIdx_;
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
    std::vector<std::weak_ptr<R2D2Actor>> partners_;

    std::vector<std::vector<float>> perCardPrivV0_;
    int noneKnown_ = 0;
    int colorKnown_ = 0;
    int rankKnown_ = 0;
    int bothKnown_ = 0;
    int testVariable_ = 0;


    // My changes
    std::unordered_map<std::string,float> stats_;
    std::vector<std::vector<std::string>> conventions_;
    int conventionIdx_;
    int conventionOverride_;
    bool conventionFictitiousOverride_;
    bool actParameterized_;
    bool recordStats_;
    bool useExperience_;
    bool logStats_;
    std::unordered_map<char, int> colourMoveToIndex_{
      {'R', 0}, {'Y', 1}, {'G', 2}, {'W', 3}, {'B', 4} };
    std::unordered_map<char, int> rankMoveToIndex_{
      {'1', 0}, {'2', 1}, {'3', 2}, {'4', 3}, {'5', 4} };
    int livesBeforeMove_;
    std::string currentTwoStep_;
    bool beliefStats_;
    bool showOwnCards_;
    bool sadLegacy_;
    bool iqlLegacy_;
    bool beliefSadLegacy_;
    bool sentSignal_;
    bool sentSignalStats_;
    bool beliefStatsSignalReceived_;
    int beliefStatsResponsePosition_;
    std::shared_ptr<hle::HanabiHand> previousHand_ = nullptr;
    bool colorShuffleSync_;
    bool colourPermuteConstant_;
    std::vector<bool> compShuffleColor_;
    std::vector<std::vector<int>> compColorPermutes_;
    std::vector<std::vector<int>> compInvColorPermutes_;

    std::vector<std::shared_ptr<rela::BatchRunner>> compRunners_;
    std::vector<rela::TensorDict> compHidden_;
    std::vector<bool> compSad_;
    std::vector<bool> compSadLegacy_;
    std::vector<bool> compIqlLegacy_;
    std::vector<bool> compHideAction_;
    std::vector<std::string> compNames_;
    std::vector<rela::FutureReply> compFutReply_;
    int partnerIdx_;
    int numPartners_;
    std::unordered_map<std::string,int> colourPermutationMap_;
    std::vector<std::vector<int>> allColourPermutations_;
    std::vector<std::vector<int>> allInvColourPermutations_;
    bool distShuffleColour_;
    std::vector<std::vector<float>> permutationDistribution_;
    std::vector<int> shuffleIndex_;
    rela::TensorDict gameStory_;
    bool shuffleConvention_;
};
