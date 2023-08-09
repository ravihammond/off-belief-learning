#include <stdio.h>
#include <algorithm>
#include <random>
#include <limits>

#include "r2d2_actor.h"
#include "rlcc/utils.h"

using namespace std;

#define PR false

void R2D2Actor::conventionReset(const HanabiEnv& env) {
  (void)env;
  if (convention_.size() == 0 || convention_[conventionIdx_].size() == 0) {
    return;
  }
  sentSignal_ = false;
  sentSignalStats_ = false;
  beliefStatsSignalReceived_ = false;
  auto responseMove = strToMove(convention_[conventionIdx_][0][1]);
  beliefStatsResponsePosition_ = responseMove.CardIndex();
}

void R2D2Actor::shadowObserve(HanabiEnv& env) {
  const auto& state = env.getHleState();

  rela::TensorDict input;
  input = observe(
      state,
      playerIdx_,
      shadowShuffleColor_.at(0),
      shadowColorPermutes_.at(0),
      shadowInvColorPermutes_.at(0),
      shadowHideAction_.at(0),
      trinary_,
      shadowSad_.at(0),
      showOwnCards_,
      shadowSadLegacy_.at(0));
  // add features such as eps and temperature
  input["eps"] = torch::tensor(playerEps_);
  if (playerTemp_.size() > 0) {
    input["temperature"] = torch::tensor(playerTemp_);
  }
  input["convention_idx"] = torch::tensor(conventionIdx_);
  input["actor_index"] = torch::tensor(playerIdx_); 

  for (size_t i = 0; i < shadowRunners_.size(); i++) {
    if (convexHull_ && convexHullWeights_.at(i) == 0) {
      continue;
    }
    addHid(input, shadowHidden_.at(i));
    shadowFutReply_.at(i) = shadowRunners_.at(i)->call("act", input);
    for (auto& kv: shadowHidden_.at(i)) {
      input.erase(kv.first);
    }
  }
}

rela::TensorDict R2D2Actor::shadowAct(const HanabiEnv& env, 
   rela::TensorDict actorReply, int curPlayer) {
  char colourMap[5] = {'R', 'Y', 'G', 'W', 'B'};
  auto game = env.getHleGame();

  vector<vector<float>> allQValues;
  vector<float> legalMoves;

  for (int i = 0; i < (int)shadowRunners_.size(); i++) {
    if (convexHull_ && convexHullWeights_.at(i) == 0) {
      allQValues.push_back(vector<float>(21, 0));
      continue;
    }
    auto reply = shadowFutReply_.at(i).get();
    moveHid(reply, shadowHidden_.at(i));

    int action = reply.at("a").item<int64_t>();
    auto move = game.GetMove(action);

    if (shadowShuffleColor_.at(i) && 
        move.MoveType() == hle::HanabiMove::Type::kRevealColor) {
      char colourBefore = colourMap[move.Color()];
      int realColor = shadowInvColorPermutes_.at(i).at(move.Color());
      move.SetColor(realColor);
      if(PR)printf("shadow action colour %c->%c\n", colourBefore, colourMap[move.Color()]);
      action = game.GetMoveUid(move);
    }
    if(PR)printf("model: %d, action: %d, %s %f\n", 
        i, action, move.ToString().c_str(), convexHullWeights_.at(i));

    if (actorReply.size() > 0 && curPlayer == playerIdx_ && logStats_) {
      compareShadowAction(env, actorReply, action);
    }

    if (convexHull_) {
      auto qValuesReply = reply.at("all_q");
      vector<float> qValues;
      for (int i = 0; i < 21; i++) {
        qValues.push_back(qValuesReply[i].item<float_t>());
      }
      allQValues.push_back(qValues);

      if (legalMoves.size() == 0) {
        auto legalMovesReply = reply.at("legal_moves");
        for (int i = 0; i < 21; i++) {
          legalMoves.push_back(legalMovesReply[i].item<float_t>());
        }
      }
    }
  }

  if (convexHull_) {
    actorReply = combineQValues(allQValues, legalMoves);
    actorReply["legal_moves"] = torch::tensor(legalMoves);
    actorReply["explore_a"] = actorReply["a"];
  }

  return actorReply;
}

void R2D2Actor::compareShadowAction(const HanabiEnv& env, 
    rela::TensorDict actorReply, int action) {
  char colourMap[5] = {'R', 'Y', 'G', 'W', 'B'};
  auto game = env.getHleGame();

  int actorAction = actorReply.at("a").item<int64_t>();
  auto actorMove = game.GetMove(actorAction);
  if (shuffleColor_ && actorMove.MoveType() == hle::HanabiMove::Type::kRevealColor) {
    char colourBefore = colourMap[actorMove.Color()];
    auto invColorPermute = &(invColorPermutes_.at(0));
    int realActorColor = (*invColorPermute).at(actorMove.Color());
    actorMove.SetColor(realActorColor);
    if(PR)printf("out actor action colour %c->%c\n", 
        colourBefore, colourMap[actorMove.Color()]);
    actorAction = game.GetMoveUid(actorMove);
  }
  
  if (actorAction == action) {
    incrementStat("turn_" + to_string(env.numStep()) + "_same");
  } else {
    incrementStat("turn_" + to_string(env.numStep()) + "_different");
  }
}

rela::TensorDict R2D2Actor::combineQValues(vector<vector<float>> allQValues,
    vector<float> legalMoves) {
  rela::TensorDict reply;
  if (allQValues.size() == 0) {
    return reply;
  }

  if (PR) {
    printf("model q values:\n");
    for (int i = 0; i < (int)allQValues.at(0).size(); i++) {
      printf("%d\t", i);
    }
    printf("\n");
    for (auto qValues: allQValues) {
      for (float qValue: qValues) {
        printf("%.4f\t", qValue);
      }
      printf("\n");
    }
  }

  vector<float> combinedQValues;

  for (int action = 0; action < (int)allQValues.at(0).size(); action++) {
    float qSum = 0;
    for (int i = 0; i < (int)allQValues.size(); i++) {
      float weight = convexHullWeights_.at(i);
      float scaledQ = allQValues.at(i).at(action) * weight;
      qSum += scaledQ;
    }
    combinedQValues.push_back(qSum);
  }

  reply["all_q"] = torch::tensor(combinedQValues);
  int actorAction = getLegalGreedyAction(combinedQValues, legalMoves);
  reply["a"] = torch::tensor(actorAction);

  return reply;
}

int R2D2Actor::getLegalGreedyAction(std::vector<float> allQValues,
    std::vector<float> legalMoves) {
  int action = 0; 
  float maxQValue = std::numeric_limits<int>::min();

  if (PR) {
    printf("combined values:\n");
    for (int i = 0; i < (int)allQValues.size(); i++) {
      printf("%d\t", i);
    }
    printf("\n");
    for (auto q: allQValues) {
      printf("%.4f\t", q);
    }
    printf("\n");
    printf("legal moves:\n");
    for (auto a: legalMoves) {
      printf("%d\t", (int)a);
    }
    printf("\n");
  }

  for (int i = 0; i < (int)allQValues.size(); i++) {
    float value = allQValues.at(i);
    int legal = (int)legalMoves.at(i);

    if (!legal) {
      continue;
    }

    if (value > maxQValue) {
      action = i;
      maxQValue = value;
    }
  }

  return action;
}

hle::HanabiMove R2D2Actor::overrideMove(const HanabiEnv& env, hle::HanabiMove move, 
    vector<float> actionQ, bool exploreAction, vector<float> legalMoves) {
  if (conventionOverride_ == 0|| convention_.size() == 0 || 
      convention_[conventionIdx_].size() == 0) {
    return move;
  }
  auto lastMove = env.getMove(env.getLastAction());
  auto signalMove = strToMove(convention_[conventionIdx_][0][0]);
  auto responseMove = strToMove(convention_[conventionIdx_][0][1]);
  auto& state = env.getHleState();
  int nextPlayer = (playerIdx_ + 1) % 2;

  if ((conventionOverride_ == 1 || conventionOverride_ == 3)
      && (lastMove.MoveType() == hle::HanabiMove::kPlay 
        || lastMove.MoveType() == hle::HanabiMove::kDiscard) 
      && sentSignal_
      && lastMove.CardIndex() <= responseMove.CardIndex()) {
    sentSignal_ = false;
  }

  if (conventionOverride_ == 2 || conventionOverride_ == 3 ) {
    if (lastMove == signalMove) {
      return responseMove;
    } else if (move == responseMove) {
      vector<hle::HanabiMove> exclude = {responseMove};
      if (conventionOverride_ == 3) {
        exclude.push_back(signalMove);
        if (!sentSignal_ 
            && movePlayableOnFireworks(env, responseMove, nextPlayer) 
            && state.MoveIsLegal(signalMove)) {
          sentSignal_ = true;
          return signalMove;
        }
      }
      return different_action(env, exclude, actionQ, exploreAction, legalMoves);
    }
  }

  if (conventionOverride_ == 1 || conventionOverride_ == 3) {
    if (!sentSignal_ && movePlayableOnFireworks(env, responseMove, nextPlayer) 
        && state.MoveIsLegal(signalMove)) {
      sentSignal_ = true;
      return signalMove;
    } else if (move == signalMove) {
      vector<hle::HanabiMove> exclude = {signalMove};
      if (conventionOverride_ == 3) exclude.push_back(responseMove);
      return different_action(env, exclude, actionQ, exploreAction, legalMoves);
    }
  }

  return move;
}

bool R2D2Actor::movePlayableOnFireworks(const HanabiEnv& env, hle::HanabiMove move, 
    int player) {
  auto& state = env.getHleState();
  hle::HanabiObservation obs = env.getObsShowCards();
  auto& allHands = obs.Hands();
  auto partnerCards = allHands[player].Cards();
  auto focusCard = partnerCards[move.CardIndex()];

  if (state.CardPlayableOnFireworks(focusCard))
    return true;

  return false;
}

hle::HanabiMove R2D2Actor::different_action(const HanabiEnv& env, 
    vector<hle::HanabiMove> exclude, vector<float> actionQ, 
    bool exploreAction, vector<float> legalMoves) {
  assert(actionQ.size() == 21);
  assert(legalMoves.size() == 21);

  auto game = env.getHleGame();

  for (auto exclude_move: exclude) {
    actionQ[game.GetMoveUid(exclude_move)] = 0;
    legalMoves[game.GetMoveUid(exclude_move)] = 0;
  }

  int nextBestMove = -1;

  if (exploreAction) {
    vector<int> legalIndices;
    vector<int> output;
    for(size_t i = 0; i < legalMoves.size(); i++) 
      if(legalMoves[i]) 
        legalIndices.push_back((int)i);
    sample(legalIndices.begin(), legalIndices.end(),
        back_inserter(output), 1, rng_);

    nextBestMove = output[0];
  } else {
    nextBestMove = distance(actionQ.begin(), max_element(
          actionQ.begin(), actionQ.end()));
  }
  assert(nextBestMove != -1);

  return game.GetMove(nextBestMove);
}

hle::HanabiMove R2D2Actor::strToMove(string key) {
  auto move = hle::HanabiMove(hle::HanabiMove::kInvalid, -1, -1, -1, -1);

  assert(key.length() == 2);
  char move_type = key[0];
  char move_target = key[1];

  switch (move_type) {
    case 'P':
      move.SetMoveType(hle::HanabiMove::kPlay);
      move.SetCardIndex(move_target - '0');
      break;
    case 'D':
      move.SetMoveType(hle::HanabiMove::kDiscard);
      move.SetCardIndex(move_target - '0');
      break;
    case 'C':
      move.SetMoveType(hle::HanabiMove::kRevealColor);
      move.SetColor(colourMoveToIndex_[move_target]);
      move.SetTargetOffset(1);
      break;
    case 'R':
      move.SetMoveType(hle::HanabiMove::kRevealRank);
      move.SetRank(rankMoveToIndex_[move_target]);
      move.SetTargetOffset(1);
      break;
    default:
      move.SetMoveType(hle::HanabiMove::kInvalid);
      break;
  }
  assert(move.MoveType() != hle::HanabiMove::kInvalid);

  return move;
}

void R2D2Actor::incrementStat(std::string key) {
  if (stats_.find(key) == stats_.end()) stats_[key] = 0;
  stats_[key]++;
}

void R2D2Actor::incrementStatsBeforeMove(
    const HanabiEnv& env, hle::HanabiMove move) {
  if (!recordStats_) {
    return;
  }

  string colours[5] = {"red", "yellow", "green", "white", "blue"};
  string ranks[5] = {"1", "2", "3", "4", "5"};

  switch(move.MoveType()) {
    case hle::HanabiMove::kPlay:
      incrementStat("play");
      incrementStat("play_" + to_string(move.CardIndex()));
      break;
    case hle::HanabiMove::kDiscard:
      incrementStat("discard");
      incrementStat("discard_" + to_string(move.CardIndex()));
      break;
    case hle::HanabiMove::kRevealColor:
      incrementStat("hint_colour");
      incrementStat("hint_" + colours[move.Color()]);
      break;
    case hle::HanabiMove::kRevealRank:
      incrementStat("hint_rank");
      incrementStat("hint_" + ranks[move.Rank()]);
      break;
    default:
      break;
  }   

  incrementStatsConvention(env, move);
  incrementStatsTwoStep(env, move);

  livesBeforeMove_ = env.getLife();
}

void R2D2Actor::incrementStatsConvention(
    const HanabiEnv& env, hle::HanabiMove move) {
  if (convention_.size() == 0 || 
      convention_[conventionIdx_].size() == 0) {
    return;
  }

  auto lastMove = env.getMove(env.getLastAction());
  auto signalMove = strToMove(convention_[conventionIdx_][0][0]);
  auto responseMove = strToMove(convention_[conventionIdx_][0][1]);
  auto& state = env.getHleState();
  bool shouldHavePlayedSignal = false;
  bool shouldHavePlayedResponse = false;
  int nextPlayer = (playerIdx_ + 1) % 2;

  // Have seen teammate play response move, or a card before response card
  if ((lastMove.MoveType() == hle::HanabiMove::kPlay
        || lastMove.MoveType() == hle::HanabiMove::kDiscard)
      && sentSignalStats_
      && lastMove.CardIndex() <= responseMove.CardIndex()) {
    sentSignalStats_ = false;
  }

  // Should play the response move
  if (lastMove == signalMove && state.MoveIsLegal(responseMove)) {
    shouldHavePlayedResponse = true;
  }

  // Should play the signal move
  if (!shouldHavePlayedResponse
      && !sentSignalStats_ 
      && movePlayableOnFireworks(env, responseMove, nextPlayer)
      && state.MoveIsLegal(signalMove)) {
    shouldHavePlayedSignal = true;
  } 

  // Signal move has been played
  if (move == signalMove) {
    sentSignalStats_ = true;
  }

  // Current turn caused a life to be lost
  if (shouldHavePlayedResponse 
      && move == responseMove 
      && !movePlayableOnFireworks(env, move, playerIdx_)) {
    incrementStat("response_played_life_lost");
  }

  incrementStatsConventionRole(shouldHavePlayedResponse, "response", move, responseMove);
  incrementStatsConventionRole(shouldHavePlayedSignal, "signal", move, signalMove);
}

void R2D2Actor::incrementStatsConventionRole(bool shouldHavePlayed, string role,
    hle::HanabiMove movePlayed, hle::HanabiMove moveRole) {
  string roleStr = role + "_" + conventionString();

  if (shouldHavePlayed) {
    incrementStat(roleStr + "_available");
  }

  if (movePlayed == moveRole) {
    incrementStat(roleStr + "_played");
    if (shouldHavePlayed) {
      incrementStat(roleStr + "_played_correct");
    } else {
      incrementStat(roleStr + "_played_incorrect");
    }
  }

}

string R2D2Actor::conventionString() {
  string conventionStr = "";

  auto conventionSet = convention_[conventionIdx_];
  for (size_t i = 0; i < conventionSet.size(); i++) {
    if (i > 0) {
      conventionStr += "-";
    }
    auto convention = conventionSet[i];
    conventionStr += convention[0] + convention[1];
  }

  return conventionStr;
}

void R2D2Actor::incrementStatsTwoStep(
    const HanabiEnv& env, hle::HanabiMove move) {
  auto lastMove = env.getMove(env.getLastAction());
  string colours[5] = {"R", "Y", "G", "W", "B"};
  string ranks[5] = {"1", "2", "3", "4", "5"};

  string stat = "";

  switch(lastMove.MoveType()) {
    case hle::HanabiMove::kRevealColor:
      stat = "C" + colours[lastMove.Color()] ;
      break;
    case hle::HanabiMove::kRevealRank:
      stat = "R" + ranks[lastMove.Rank()];
      break;
    case hle::HanabiMove::kPlay:
      stat = "P" + to_string(lastMove.CardIndex());
      break;
    case hle::HanabiMove::kDiscard:
      stat = "D" + to_string(lastMove.CardIndex());
      break;
    default:
      currentTwoStep_ = "X";
      return;
  }

  incrementStat(stat);

  switch(move.MoveType()) {
    case hle::HanabiMove::kPlay:
      stat += "_P" + to_string(move.CardIndex());
      break;
    case hle::HanabiMove::kDiscard:
      stat += "_D" + to_string(move.CardIndex());
      break;
    case hle::HanabiMove::kRevealColor:
      stat += "_C" + colours[move.Color()];
      break;
    case hle::HanabiMove::kRevealRank:
      stat += "_R" + ranks[move.Rank()];
      break;
    default:
      currentTwoStep_ = "X";
      return;
  }

  currentTwoStep_ = stat;
  incrementStat(stat);
}


void R2D2Actor::incrementStatsAfterMove(
    const HanabiEnv& env) {
  if (!recordStats_) {
    return;
  }

  if (env.getCurrentPlayer() != playerIdx_ &&
      env.getLife() != livesBeforeMove_ &&
      currentTwoStep_ != "X") {
    incrementStat("dubious_" + currentTwoStep_);
  }
}

tuple<int, int, vector<int>> R2D2Actor::beliefConventionPlayable(const HanabiEnv& env) {
  int curPlayer = env.getCurrentPlayer();
  vector<int> playableCards(25, 0);
  assert(!partners_[(playerIdx_ + 1) % 2].expired());
  auto partner = partners_[(playerIdx_ + 1) % 2].lock();
  if (curPlayer != playerIdx_ 
      || convention_.size() == 0 
      || convention_[conventionIdx_].size() == 0
      || partner->previousHand_ == nullptr) {
    return make_tuple(0, beliefStatsResponsePosition_, playableCards);
  }

  auto& state = env.getHleState();
  auto partnerLastMove = env.getMove(env.getLastAction());
  auto myLastMove = env.getMove(env.getSecondLastAction());
  auto signalMove = strToMove(convention_[conventionIdx_][0][0]);
  auto responseMove = strToMove(convention_[conventionIdx_][0][1]);

  // Reset if partners play may be the signal card 
  if (beliefStatsSignalReceived_
      && partnerLastMove.MoveType() == hle::HanabiMove::kPlay
      && playedCardPossiblySignalledCard(
        partnerLastMove, partner->previousHand_)) {
    beliefStatsSignalReceived_ = false;
    beliefStatsResponsePosition_ = responseMove.CardIndex();
  }

  // Reset or shift position if my last action was discard
  if (beliefStatsSignalReceived_
      && myLastMove.MoveType() == hle::HanabiMove::kDiscard) {
    if (myLastMove.CardIndex() < beliefStatsResponsePosition_) {
      beliefStatsResponsePosition_--;
    } else if (myLastMove.CardIndex() == beliefStatsResponsePosition_) {
      beliefStatsSignalReceived_ = false;
      beliefStatsResponsePosition_ = responseMove.CardIndex();
    }
  }

  // Reset or shift position if my last action was play
  if (beliefStatsSignalReceived_
      && myLastMove.MoveType() == hle::HanabiMove::kPlay) {
    if (myLastMove.CardIndex() == beliefStatsResponsePosition_) {
      beliefStatsSignalReceived_ = false;
      beliefStatsResponsePosition_ = responseMove.CardIndex();
    } else if (previousHand_ != nullptr
        && playedCardPossiblySignalledCard(myLastMove, previousHand_)) {
      beliefStatsSignalReceived_ = false;
      beliefStatsResponsePosition_ = responseMove.CardIndex();
    } else if (myLastMove.CardIndex() < beliefStatsResponsePosition_) {
      beliefStatsResponsePosition_--;
    }
  }

  if (partnerLastMove == signalMove) {
    beliefStatsSignalReceived_ = true;
  }

  auto obs = env.getObsShowCards();
  auto& all_hands = obs.Hands();
  auto myHand = all_hands[playerIdx_];
  auto conventionCard = myHand.Cards()[beliefStatsResponsePosition_];

  if (beliefStatsSignalReceived_) {
    possibleResponseCards(env, playableCards);
    incrementStat("response_should_be_playable");
    if (state.CardPlayableOnFireworks(conventionCard)) {
      incrementStat("response_is_playable");
    }
    return make_tuple(1, beliefStatsResponsePosition_, playableCards);
  }

  return make_tuple(0, beliefStatsResponsePosition_, playableCards);
}

bool R2D2Actor::playedCardPossiblySignalledCard(hle::HanabiMove playedMove,
    shared_ptr<hle::HanabiHand> playedHand) {
  auto myHandKnowledge = previousHand_->Knowledge();
  auto signalledCardKnowledge = myHandKnowledge[beliefStatsResponsePosition_];
  string colourKnowledgeStr = signalledCardKnowledge.ColorKnowledgeRangeString();
  string rankKnowledgeStr = signalledCardKnowledge.RankKnowledgeRangeString();

  auto playedCard = playedHand->Cards()[playedMove.CardIndex()];
  char colours[5] = {'R','Y','G','W','B'};
  char ranks[5] = {'1','2','3','4','5'};
  char playedColour = colours[playedCard.Color()];
  char playedRank = ranks[playedCard.Rank()];

  bool colourMatch = false;
  bool rankMatch = false;

  for (char& ch: colourKnowledgeStr) {
    if (ch == playedColour) {
      colourMatch = true;
      break;
    }
  }

  for (char& ch: rankKnowledgeStr) {
    if (ch == playedRank) {
      rankMatch = true;
      break;
    }
  }

  return colourMatch && rankMatch;
}

void R2D2Actor::possibleResponseCards(const HanabiEnv& env, 
    vector<int>& playableCards) {
  auto obs = env.getObsShowCards();
  auto& all_hands = obs.Hands();
  auto cards = all_hands[playerIdx_].Cards();
  auto game = env.getHleGame();
  auto& state = env.getHleState();

  for (auto card: cards) {
    int id = game.CardToIndex(card.Value());
    if (state.CardPlayableOnFireworks(card)) {
      playableCards[id] = 1;
    }
  }

  auto deck = state.Deck();

  for (int colour = 0; colour < 5; colour++) {
    for (int rank = 0; rank < 5; rank++) {
      int id = colour * 5 + rank;
      auto cardValue = indexToCard(id, game.NumRanks());
      auto card = hle::HanabiCard(
          cardValue, -1);
      if (deck.CardCount(colour, rank) > 0
          && state.CardPlayableOnFireworks(card)) {
        playableCards[id] = 1;
      }
    }
  }
}

