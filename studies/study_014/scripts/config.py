import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from study_utils import BaseStudyConfig, PromptBuilder

import random


class AuctionPromptBuilder(PromptBuilder):
    def build_trial_prompt(self, trial_metadata):
        profile = trial_metadata.get("profile", {})
        current_round = trial_metadata.get("current_round", 1)
        num_bidders = trial_metadata.get("num_bidders", 1)
        history = trial_metadata.get("history", [])
        instructions_text = trial_metadata.get("instructions_text", "")

        full_prompt = []

        full_prompt.append(f"You are an undergraduate student at the University of Michigan.")
        full_prompt.append(f"You are {profile.get('age', 20)}, {profile.get('gender', 'Male')}, "
                           f"{profile.get('race', '')}, and studying {profile.get('program', '')}.\n")

        full_prompt.append("You are about to participate in an auction experiment.\n")
        full_prompt.append(f"Here are the experiment instructions:\n{instructions_text}\n")

        full_prompt.append("IMPORTANT:")
        full_prompt.append("- Try to maximize your total profit over 60 rounds.")
        full_prompt.append("- You can only respond with an integer between 0 and 100 representing the reservation price.")
        full_prompt.append("- Do not provide any explanation or additional text in your response.\n")

        # Last round result
        if history:
            last = history[-1]
            bid_prices = last.get("bid_prices", [])
            reserve = last.get("reserve_price", 0)
            filtered_bids = sorted([b for b in bid_prices if b >= reserve], reverse=True)

            full_prompt.append(f"Here is your last round result:")
            full_prompt.append(f"Round {last['round']}:")
            full_prompt.append(f"  Your Reserve Price: {reserve}")
            full_prompt.append(f"  Profit: {last['profit']}")
            full_prompt.append(f"  Number of Bidders: {last['num_bidders']}")
            full_prompt.append(f"  Winning Price: {last['profit']}")
            if filtered_bids:
                bids_str = "; ".join([f"Bidder: {i+1}, Drop-Out Price: {p}" for i, p in enumerate(filtered_bids)])
            else:
                bids_str = "No bids were above your reserve price."
            full_prompt.append(f"  Bids higher than your reserve price: {bids_str}\n")
        else:
            full_prompt.append("Here is your last round result:\nNo previous round result.\n")

        # Cumulative history
        if history:
            full_prompt.append("Here is the history of all previous rounds:")
            for h in history:
                drop_out = [p if p != 0 else None for p in h["bid_prices"]]
                full_prompt.append(
                    f"Round {h['round']}: "
                    f"Your Reserve Price={h['reserve_price']}, "
                    f"Your Profit={h['profit']}, "
                    f"Number of Bidders={h['num_bidders']}, "
                    f"Drop-Out Prices={drop_out}"
                )
            full_prompt.append("")
        else:
            full_prompt.append("Here is the history of all previous rounds:\nNo prior history.\n")

        full_prompt.append(f"Now it's round {current_round}.")
        full_prompt.append(f"Number of Bidders in this round: {num_bidders}\n")
        full_prompt.append("What reserve price do you set for *this round*?")

        full_prompt.append("\nRESPONSE_SPEC (MANDATORY FORMAT):")
        full_prompt.append("- Output ONLY a single integer between 0 and 100.")
        full_prompt.append("- Expected lines: 1")

        return "\n".join(full_prompt)


class StudyDavisEtAl2011Config(BaseStudyConfig):
    prompt_builder_class = AuctionPromptBuilder
    REQUIRES_GROUP_TRIALS = True

    def _parse_bid_prices(self, bid_string):
        return [int(x) for x in re.findall(r'd:(\d+)', str(bid_string))]

    def create_trials(self, n_trials=None):
        spec = self.load_specification()
        materials = self.load_material("auction_instructions")
        instructions_text = materials["instructions"]

        n_subjects = n_trials if n_trials is not None else spec["participants"]["n"]

        # Load human auction data for bidder group structure
        benchmark_path = self.study_path / "benchmark" / "human_data" / "auction_human_data.csv"
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Human data not found: {benchmark_path}")

        import pandas as pd
        human_data = pd.read_csv(benchmark_path)
        bidder_groups = sorted(human_data["Bidder Group"].unique())

        trials = []
        for bg in bidder_groups[:n_subjects]:
            group_data = human_data[human_data["Bidder Group"] == bg].reset_index(drop=True)

            rounds = []
            for i, row in group_data.iterrows():
                rounds.append({
                    "round": i + 1,
                    "num_bidders": int(row["numBidders"]),
                    "bid_prices": self._parse_bid_prices(row["newBids"])
                })

            profile = {
                "age": random.randint(18, 22),
                "gender": random.choice(["Male", "Female"]),
                "race": random.choice(["White", "Asian", "Black", "Hispanic"]),
                "program": random.choice(["Economics", "Psychology", "Engineering", "Business"])
            }

            trials.append({
                "sub_study_id": "second_price_auction",
                "bidder_group": bg,
                "rounds": rounds,
                "profile": profile,
                "instructions_text": instructions_text
            })

        return trials

    def aggregate_results(self, raw_results):
        aggregated = {"by_bidder_count": {}, "all_observations": []}

        for record in raw_results.get("individual_data", []):
            response_text = record.get("response_text", "")
            trial_info = record.get("trial_info", {})
            num_bidders = trial_info.get("num_bidders")

            match = re.search(r"(\d+)", str(response_text))
            if not match:
                continue
            reserve_price = int(match.group(1))
            if reserve_price < 0 or reserve_price > 100:
                continue

            aggregated["all_observations"].append({
                "reserve_price": reserve_price,
                "num_bidders": num_bidders
            })

            key = str(num_bidders)
            if key not in aggregated["by_bidder_count"]:
                aggregated["by_bidder_count"][key] = []
            aggregated["by_bidder_count"][key].append(reserve_price)

        # Compute summary statistics
        final = {"descriptive_statistics": {}, "inferential_statistics": []}

        for bc, prices in aggregated["by_bidder_count"].items():
            final["descriptive_statistics"][f"mean_rPrice_{bc}_bidders"] = float(np.mean(prices))
            final["descriptive_statistics"][f"sd_rPrice_{bc}_bidders"] = float(np.std(prices))
            final["descriptive_statistics"][f"n_{bc}_bidders"] = len(prices)

        all_obs = aggregated["all_observations"]
        if len(all_obs) >= 10:
            rprices = np.array([o["reserve_price"] for o in all_obs])
            nbidders = np.array([o["num_bidders"] for o in all_obs])
            r = np.corrcoef(rprices, nbidders)[0, 1]
            final["inferential_statistics"].append({
                "test": "pearson_correlation",
                "variables": ["reserve_price", "num_bidders"],
                "statistic": float(r),
                "n": len(all_obs)
            })

        return final
