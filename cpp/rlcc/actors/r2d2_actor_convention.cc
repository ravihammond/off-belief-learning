#include <stdio.h>
#include <algorithm>
#include <random>

#include "r2d2_actor.h"
#include "rlcc/utils.h"

using namespace std;

#define CV true

void R2D2Actor::conventionReset(const HanabiEnv& env) {
  (void)env;
  if (conventions_.size() == 0 || conventions_[conventionIdx_].size() == 0) {
    return;
  }
  sentSignal_ = false;
  sentSignalStats_ = false;
  beliefStatsSignalReceived_ = false;
  auto responseMove = strToMove(conventions_[conventionIdx_][1]);
  beliefStatsResponsePosition_ = responseMove.CardIndex();
}

void R2D2Actor::callCompareAct(HanabiEnv& env) {
  const auto& state = env.getHleState();
  for (size_t i = 0; i < compRunners_.size(); i++) {
    rela::TensorDict input;
    input = observe(
        state,
        playerIdx_,
        compShuffleColor_.at(i),
        compColorPermutes_.at(i),
        compInvColorPermutes_.at(i),
        compHideAction_.at(i),
        trinary_,
        compSad_.at(i),
        showOwnCards_,
        compSadLegacy_.at(i),
        compIqlLegacy_.at(i));
    // add features such as eps and temperature
    input["eps"] = torch::tensor(playerEps_);
    if (playerTemp_.size() > 0) {
      input["temperature"] = torch::tensor(playerTemp_);
    }
    input["convention_idx"] = torch::tensor(conventionIdx_);
    input["actor_index"] = torch::tensor(playerIdx_); 
    addHid(input, compHidden_[i]);
    compFutReply_[i] = compRunners_[i]->call("act", input);
  }
}

void R2D2Actor::replyCompareAct(const HanabiEnv& env, 
    int actorAction, int curPlayer) {
  char colourMap[5] = {'R', 'Y', 'G', 'W', 'B'};
  auto game = env.getHleGame();
  for (size_t i = 0; i < compRunners_.size(); i++) {
    auto reply = compFutReply_[i].get();
    moveHid(reply, compHidden_[i]);

    int action = reply.at("a").item<int64_t>();
    auto move = game.GetMove(action);

    if (compShuffleColor_[i] && 
        move.MoveType() == hle::HanabiMove::Type::kRevealColor) {
      char colourBefore = colourMap[move.Color()];
      int realColor = compInvColorPermutes_[i].at(move.Color());
      move.SetColor(realColor);
      if(CV)printf("shadow action colour %c->%c\n", colourBefore, colourMap[move.Color()]);
      action = game.GetMoveUid(move);
    }

    auto actorMove = game.GetMove(actorAction);
    if (shuffleColor_ && actorMove.MoveType() == hle::HanabiMove::Type::kRevealColor) {
      char colourBefore = colourMap[actorMove.Color()];
      auto invColorPermute = &(invColorPermutes_[0]);
      int realActorColor = (*invColorPermute)[actorMove.Color()];
      actorMove.SetColor(realActorColor);
      if(CV)printf("out actor action colour %c->%c\n", 
          colourBefore, colourMap[actorMove.Color()]);
      actorAction = game.GetMoveUid(actorMove);
    }
    
    if (curPlayer != playerIdx_) {
      continue;
    }

    if (actorAction == action) {
      incrementStat("turn_" + to_string(env.numStep()) + "_same");
    } else {
      incrementStat("turn_" + to_string(env.numStep()) + "_different");
    }
  }
}

hle::HanabiMove R2D2Actor::overrideMove(const HanabiEnv& env, hle::HanabiMove move, 
    vector<float> actionQ, bool exploreAction, vector<float> legalMoves) {
  if (conventionOverride_ == 0|| conventions_.size() == 0 || 
      conventions_[conventionIdx_].size() == 0) {
    return move;
  }
  auto lastMove = env.getMove(env.getLastAction());
  auto signalMove = strToMove(conventions_[conventionIdx_][0]);
  auto responseMove = strToMove(conventions_[conventionIdx_][1]);
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
      if(CV)printf("Original move: %s\n", move.ToString().c_str());
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
      
      auto move = different_action(env, exclude, actionQ, exploreAction, legalMoves);
      if(CV)printf("Original move: %s\n", move.ToString().c_str());
      return move;
    }
  }

  if (conventionOverride_ == 1 || conventionOverride_ == 3) {
    if (!sentSignal_ && movePlayableOnFireworks(env, responseMove, nextPlayer) 
        && state.MoveIsLegal(signalMove)) {
      sentSignal_ = true;
      if(CV)printf("Original move: %s\n", move.ToString().c_str());
      return signalMove;
    } else if (move == signalMove) {
      vector<hle::HanabiMove> exclude = {signalMove};
      if (conventionOverride_ == 3) exclude.push_back(responseMove);
      auto move = different_action(env, exclude, actionQ, exploreAction, legalMoves);
      if(CV)printf("Original move: %s\n", move.ToString().c_str());
      return move;
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
  if (stats_.find(key) == stats_.end()) {
    stats_[key] = 0;
  } 
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
  if (conventions_.size() == 0 || 
      conventions_[conventionIdx_].size() == 0) {
    return;
  }

  auto lastMove = env.getMove(env.getLastAction());
  auto signalMove = strToMove(conventions_[conventionIdx_][0]);
  auto responseMove = strToMove(conventions_[conventionIdx_][1]);
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
  auto convention = conventions_[conventionIdx_];
  return convention[0] + convention[1];
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
      || conventions_.size() == 0 
      || conventions_[conventionIdx_].size() == 0
      || partner->previousHand_ == nullptr) {
    return make_tuple(0, beliefStatsResponsePosition_, playableCards);
  }

  auto& state = env.getHleState();
  auto partnerLastMove = env.getMove(env.getLastAction());
  auto myLastMove = env.getMove(env.getSecondLastAction());
  auto signalMove = strToMove(conventions_[conventionIdx_][0]);
  auto responseMove = strToMove(conventions_[conventionIdx_][1]);

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

