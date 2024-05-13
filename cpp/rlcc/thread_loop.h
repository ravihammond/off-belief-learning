// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <stdio.h>
#include <iostream>
#include <time.h>

#include "rela/thread_loop.h"
#include "rlcc/actors/r2d2_actor.h"

#define PR true

class HanabiThreadLoop : public rela::ThreadLoop {
 public:
  HanabiThreadLoop(
      std::vector<std::shared_ptr<HanabiEnv>> envs,
      std::vector<std::vector<std::shared_ptr<R2D2Actor>>> actors,
      bool eval,
      int threadIdx,
      std::vector<std::vector<hle::HanabiCardValue>> forcedDecks)
    : envs_(std::move(envs))
      , actors_(std::move(actors))
      , done_(envs_.size(), -1)
      , eval_(eval) 
      , avgN_(400) 
      , threadIdx_(threadIdx)
      , forcedDecks_(forcedDecks) {
        assert(envs_.size() == actors_.size());
      }

  virtual void mainLoop() override {
    clock_t t;
    int game_turn = 0;
    bool returning = false;
    while (!terminated()) {
      if(PR)printf("\n=======================================\n");
      if(PR)printf("Game Turn: %d\n", game_turn);
      // go over each envs in sequential order
      // call in seperate for-loops to maximize parallization

      t = clock();
      for (size_t i = 0; i < envs_.size(); ++i) {
        if (done_[i] == 1) {
          continue;
        }

        auto& actors = actors_[i];

        if (envs_[i]->terminated()) {
          // we only run 1 game for evaluation
          if (eval_) {
            ++done_[i];
            if (done_[i] == 1) {
              numDone_ += 1;
              if (numDone_ == (int)envs_.size()) {
                returning = true;
              }
            }
          }

          if (forcedDecks_.size() > 0) {
            envs_[i]->resetWithDeck(forcedDecks_[i]);
          } else {
            envs_[i]->reset();
          }
          for (size_t j = 0; j < actors.size(); ++j) {
            if(PR)printf("\n[player %ld resetting]\n", j);
            actors[j]->reset(*envs_[i]);
          }
        }
      }

      if (returning) {
        for (size_t i = 0; i < envs_.size(); ++i) {
          auto& actors = actors_[i];
          for (size_t j = 0; j < actors.size(); ++j) {
            actors[j]->pushToReplayBuffer();
          }
        }
        return;
      }

      t = clock() - t;
      timeStats_[0] = approxRollingAverage(
          timeStats_[0], ((double)t)/CLOCKS_PER_SEC);

      for (size_t i = 0; i < envs_.size(); ++i) {
        if(PR)printf("\nScore: %d\n", envs_[i]->getScore());
        if(PR)printf("Lives: %d\n", envs_[i]->getLife());
        if(PR)printf("Information: %d\n", envs_[i]->getInfo());
        auto deck = envs_[i]->getHleState().Deck();
        if(PR)printf("Deck: %d\n", deck.Size());
        std::string colours = "RYGWB";
        if(PR)printf("Discards: ");
        auto& discards = envs_[i]->getHleState().DiscardPile();
        for (auto card: discards) {
          int card_colour = card.Color();
          int rank = card.Rank() + 1;
          if(PR)printf("%C%d ", colours[card_colour], rank);
        }
        if(PR)printf("\n");
        auto fireworks = envs_[i]->getFireworks();
        if(PR)printf("Fireworks: ");
        for (unsigned long i = 0; i < colours.size(); i++)
          if(PR)printf("%c%d ", colours[i], fireworks[i]);
        if(PR)printf("\n");
        auto hands = envs_[i]->getHleState().Hands();
        int cp = envs_[i]->getCurrentPlayer();
        for(unsigned long i = 0; i < hands.size(); i++) {
          if(PR)printf("Actor %ld hand:%s\n", i,
              cp == (int)i ? " <-- current player" : ""); 
          auto hand = hands[i].ToString();
          hand.pop_back();
          if(PR)printf("%s\n", hand.c_str());
          /* if (PR && cp == (int)i) { */
          /*   printf("e\n"); */
          /*   printf("f\n"); */
          /*   printf("g\n"); */
          /*   auto legalMoves = envs_[i]->getHleState().LegalMoves(i); */
          /*   printf("h\n"); */
          /*   std::vector<int> vLegalMove(envs_[i]->getHleGame().MaxMoves() + 1); */
          /*   printf("i\n"); */
          /*   for (auto move : legalMoves) { */
          /*     printf("j\n"); */
          /*     auto uid = envs_[i]->getHleGame().GetMoveUid(move); */
          /*     printf("i\n"); */
          /*     vLegalMove[uid] = 1; */
          /*     printf("k\n"); */
          /*   } */
          /*   printf("l\n"); */
          /*   if (legalMoves.size() == 0) { */
          /*     printf("m\n"); */
          /*     vLegalMove[envs_[i]->getHleGame().MaxMoves()] = 1; */
          /*     printf("n\n"); */
          /*   } */
          /*   printf("o\n"); */
          /*   printf("Legal moves: "); */
          /*   for (auto moveIsLegal: vLegalMove) { */
          /*     printf("%d ", moveIsLegal); */
          /*   } */
          /*   printf("\n"); */
          /* } */
        }
      }

      if(PR)printf("\n----\n");

      // go over each envs in sequential order
      // call in seperate for-loops to maximize parallization
      t = clock();
      for (size_t i = 0; i < envs_.size(); ++i) {
        if (done_[i] == 1) {
          continue;
        }

        auto& actors = actors_[i];
        int curPlayer = envs_[i]->getCurrentPlayer();
        for (size_t j = 0; j < actors.size(); ++j) {
          if(PR) printf("\n[player %ld observe before acting]%s\n", j,
              curPlayer == (int)j ? " <-- current player" : "");
          actors[j]->observeBeforeAct(*envs_[i]);
        }
      }
      t = clock() - t;
      timeStats_[1] = approxRollingAverage(
          timeStats_[1], ((double)t)/CLOCKS_PER_SEC);

      if(PR)printf("\n----\n");

      t = clock();
      for (size_t i = 0; i < envs_.size(); ++i) {
        if (done_[i] == 1) {
          continue;
        }

        auto& actors = actors_[i];
        int curPlayer = envs_[i]->getCurrentPlayer();
        for (size_t j = 0; j < actors.size(); ++j) {
          if(PR)printf("\n[player %ld acting]%s\n", j, 
              curPlayer == (int)j ? " <-- current player" : "");
          actors[j]->act(*envs_[i], curPlayer);
        }
      }
      t = clock() - t;
      timeStats_[2] = approxRollingAverage(
          timeStats_[2], ((double)t)/CLOCKS_PER_SEC);

      if(PR)printf("\n----\n");

      t = clock();
      for (size_t i = 0; i < envs_.size(); ++i) {
        if (done_[i] == 1) {
          continue;
        }

        auto& actors = actors_[i];
        int curPlayer = envs_[i]->getCurrentPlayer();
        for (size_t j = 0; j < actors.size(); ++j) {
          if(PR)printf("\n[player %ld fictious acting]%s\n", j,
              curPlayer == (int)j ? " <-- current player" : "");
          actors[j]->fictAct(*envs_[i]);
        }
      }
      t = clock() - t;
      timeStats_[3] = approxRollingAverage(
          timeStats_[3], ((double)t)/CLOCKS_PER_SEC);
      if(PR)printf("\n----\n");

      t = clock();
      for (size_t i = 0; i < envs_.size(); ++i) {
        if (done_[i] == 1) {
          continue;
        }

        auto& actors = actors_[i];
        int curPlayer = envs_[i]->getCurrentPlayer();
        for (size_t j = 0; j < actors.size(); ++j) {
          if(PR)printf("\n[player %ld observe after acting]%s\n", j,
              curPlayer == (int)j ? " <-- current player" : "");
          actors[j]->observeAfterAct(*envs_[i]);
        }
      }
      t = clock() - t;
      timeStats_[4] = approxRollingAverage(
          timeStats_[4], ((double)t)/CLOCKS_PER_SEC);
      game_turn = envs_[0]->numStep();
    }
  }

  double approxRollingAverage(double avg, double new_sample) {
    avg -= avg / avgN_;
    avg += new_sample / avgN_;
    return avg;
  }

  std::vector<double> getTimeStats() {
    return timeStats_;
  }

 private:
  std::vector<std::shared_ptr<HanabiEnv>> envs_;
  std::vector<std::vector<std::shared_ptr<R2D2Actor>>> actors_;
  std::vector<int8_t> done_;
  const bool eval_;
  int numDone_ = 0;
  std::vector<double> timeStats_ = std::vector<double>(5, 0);
  double avgN_;
  int threadIdx_;
  std::vector<std::vector<hle::HanabiCardValue>> forcedDecks_;
};

