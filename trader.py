# Optimized trader.py for Tutorial Round - V3 (Visualizer Logging Integrated)

import json
from typing import Dict, List, Any, Tuple
import jsonpickle
import numpy as np
import math
import statistics

# Assume datamodel.py containing Listing, OrderDepth, Trade, TradingState, Order etc. is available
# Ensure you have the correct datamodel.py file in your execution environment
from datamodel import * # Make sure datamodel.py is accessible

# --- Visualizer Logger Class ---
# (Copied directly from the prerequisites provided)
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750 # Slightly reduced from example for safety margin

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        # Appends logs to an internal buffer; limit checked during flush
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # Calculate base length without large variable data
        compressed_state_base = [state.timestamp, "", self.compress_listings(state.listings), self.compress_order_depths(state.order_depths), self.compress_trades(state.own_trades), self.compress_trades(state.market_trades), state.position, self.compress_observations(state.observations)]
        base_json_part = self.to_json([compressed_state_base, self.compress_orders(orders), conversions, "", ""]) # Empty strings for traderData and logs
        base_length = len(base_json_part) - 4 # Account for the 4 empty strings "" we used as placeholders

        # Calculate remaining length and divide by 3 for the variable parts
        available_length = self.max_log_length - base_length
        max_item_length = available_length // 3
        if max_item_length < 0: max_item_length = 0 # Ensure non-negative length

        # Prepare the final list with potentially truncated data
        log_entry = [
            self.compress_state(state, self.truncate(state.traderData, max_item_length)), # Compress state with potentially truncated internal traderData
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length), # Use potentially truncated output trader_data
            self.truncate(self.logs, max_item_length)    # Use potentially truncated custom logs
        ]

        # Print the single JSON line for this iteration
        print(self.to_json(log_entry))

        # Reset internal log buffer for the next iteration
        self.logs = ""

    # --- Compression Methods (Copied from Logger Prerequisites) ---
    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        # Ensure observations exist, create empty if not
        obs = state.observations if state.observations else Observation({}, {})
        return [
            state.timestamp,
            trader_data, # Use the (potentially truncated) trader_data passed in
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades if state.own_trades else {}), # Handle None trades
            self.compress_trades(state.market_trades if state.market_trades else {}), # Handle None trades
            state.position if state.position else {}, # Handle None position
            self.compress_observations(obs), # Use ensured Observation object
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        # Handle cases where listings might be None or empty
        if listings:
            for listing in listings.values():
                compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        if order_depths:
            for symbol, order_depth in order_depths.items():
                # Ensure buy_orders and sell_orders exist
                buy_orders = order_depth.buy_orders if hasattr(order_depth, 'buy_orders') and order_depth.buy_orders else {}
                sell_orders = order_depth.sell_orders if hasattr(order_depth, 'sell_orders') and order_depth.sell_orders else {}
                compressed[symbol] = [buy_orders, sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        if trades:
            for arr in trades.values():
                # Ensure arr is iterable and contains Trade objects
                if isinstance(arr, list):
                    for trade in arr:
                         # Check if trade is a Trade object before accessing attributes
                         if isinstance(trade, Trade):
                             compressed.append(
                                 [
                                     trade.symbol,
                                     trade.price,
                                     trade.quantity,
                                     trade.buyer if trade.buyer else "", # Use empty string for None
                                     trade.seller if trade.seller else "", # Use empty string for None
                                     trade.timestamp,
                                 ]
                             )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
         # Ensure nested dictionaries exist within the Observation object
         plain_obs = observations.plainValueObservations if hasattr(observations, 'plainValueObservations') and observations.plainValueObservations else {}
         conv_obs_orig = observations.conversionObservations if hasattr(observations, 'conversionObservations') and observations.conversionObservations else {}

         conversion_observations_compressed = {}
         if conv_obs_orig:
            for product, observation in conv_obs_orig.items():
                # Check if observation is a ConversionObservation object
                 if isinstance(observation, ConversionObservation):
                     conversion_observations_compressed[product] = [
                         # Ensure attributes exist before accessing, provide defaults if necessary
                         getattr(observation, 'bidPrice', 0),
                         getattr(observation, 'askPrice', 0),
                         getattr(observation, 'transportFees', 0),
                         getattr(observation, 'exportTariff', 0),
                         getattr(observation, 'importTariff', 0),
                         # Use getattr for potentially missing attributes with default 0
                         getattr(observation, 'sunlight', 0), # Adjusted field name based on datamodel
                         getattr(observation, 'humidity', 0), # Adjusted field name based on datamodel
                     ]

         return [plain_obs, conversion_observations_compressed]


    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        if orders:
            for arr in orders.values():
                 if isinstance(arr, list):
                    for order in arr:
                        if isinstance(order, Order):
                             compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        # Use ProsperityEncoder if provided and needed, else default json
        # Assuming ProsperityEncoder handles specific object types if necessary
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        # Ensure value is a string before checking length
        value_str = str(value) if value is not None else ""
        if len(value_str) <= max_length:
            return value_str
        # Truncate and add ellipsis
        return value_str[: max_length - 3] + "..."

# Create a global instance of the logger
logger = Logger()

# --- Tunable Parameters (Using V3/V1 reverted values) ---
PARAMS = {
    "shared": {
        "take_profit_threshold": 0.4, # Reverted: More aggressive sniping
        "max_history_length": 90 # Reverted: Longer history for smoother stats
    },
    "RAINFOREST_RESIN": {
        "fair_value_anchor": 10000.0,
        "anchor_blend_alpha": 0.08, # Reverted: Less weight on market WMP/EMA
        "min_spread": 7, #
        "volatility_spread_factor": 0.32, # Reverted: V1 value
        "inventory_skew_factor": 0.01, # Reverted: V1 value
        "base_order_qty": 25,      # Reverted: V1 value
        "reversion_threshold": 2   # Reverted: V1 value
    },
    "KELP": {
        "ema_alpha": 0.05,          # Reverted: Slower EMA
        "min_spread": 2,
        "volatility_spread_factor": 1.2, # Reverted: Higher sensitivity
        "inventory_skew_factor": 0.015,  # Reverted: Higher skew sensitivity
        "base_order_qty": 28,        # Reverted: Smaller base qty
        "min_volatility_qty_factor": 1.1, # Reverted: Less aggressive qty reduction
        "max_volatility_for_qty_reduction": 4.0, # Reverted
        "imbalance_depth": 5,
        "imbalance_fv_adjustment_factor": 0.36 # Reverted
    }
}

# Tutorial Round Position Limits
POSITION_LIMITS = {'RAINFOREST_RESIN': 50, 'KELP': 50}

class Trader:

    def __init__(self):
        # You can add initialization here if needed, but state should go in traderData
        # Using logger.print here might be useful for one-time init messages
        logger.print("Initializing Optimized Tutorial Trader V3 (Visualizer Enabled)...")

    # --- State Management Methods (Keep deserialize/serialize from V3) ---
    def deserialize_state(self, traderData: str) -> Dict[str, Any]:
        # (Keep the robust deserialize_state from previous version)
        default_state = {
            "price_history": {}, "ema_prices": {}, "std_devs": {},
            "fair_values": {}, "last_wmp": {}
        }
        for prod in POSITION_LIMITS:
            default_state["price_history"].setdefault(prod, [])
            default_state["ema_prices"].setdefault(prod, None)
            default_state["std_devs"].setdefault(prod, 0.0)
            default_state["fair_values"].setdefault(prod, None)
            default_state["last_wmp"].setdefault(prod, None)

        if not traderData: return default_state
        try:
            state = jsonpickle.decode(traderData)
            if not isinstance(state, dict):
                 logger.print("Warning: Decoded traderData is not a dict. Resetting state.")
                 return default_state
            for key, default_val in default_state.items():
                if key not in state or not isinstance(state[key], type(default_val)):
                    state[key] = default_val
                if isinstance(default_val, dict):
                    if not isinstance(state[key], dict): state[key] = {}
                    for prod in POSITION_LIMITS:
                         if prod not in state[key] or not isinstance(state[key].get(prod), type(default_val.get(prod))):
                              state[key][prod] = default_val.get(prod)
            return state
        except Exception as e:
            logger.print(f"Error deserializing traderData: {e}. Returning default state.")
            return default_state

    def serialize_state(self, state_dict: Dict[str, Any]) -> str:
        """Saves state to a string using jsonpickle."""
        try:
            # Using unpicklable=False is generally safer and sufficient if only standard types are used
            return jsonpickle.encode(state_dict, unpicklable=False)
        except Exception as e:
            logger.print(f"Error serializing state: {e}")
            return "" # Return empty string on error


    # --- Helper Methods (Keep calculate_*, update_ema, update_history_and_stats from V3) ---
    # ... (Keep the existing helper methods calculate_weighted_mid_price,
    #      calculate_order_book_imbalance, update_ema, update_history_and_stats,
    #      calculate_fair_value unchanged from the previous V3 code) ...

    # Make sure helper methods are correctly indented within the Trader class
    def calculate_weighted_mid_price(self, order_depth: OrderDepth) -> float | None:
        # (Same implementation as V3)
        if not order_depth.sell_orders or not order_depth.buy_orders: return None
        sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)
        sorted_asks = sorted(order_depth.sell_orders.items())
        best_bid_price, best_bid_vol = sorted_bids[0]
        best_ask_price, best_ask_vol = sorted_asks[0]
        best_ask_vol = abs(best_ask_vol)
        if best_bid_price >= best_ask_price: return (best_bid_price + best_ask_price) / 2.0
        if best_bid_vol == 0 and best_ask_vol == 0: return (best_bid_price + best_ask_price) / 2.0
        if best_bid_vol == 0: return best_ask_price
        if best_ask_vol == 0: return best_bid_price
        return (best_bid_price * best_ask_vol + best_ask_price * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def calculate_order_book_imbalance(self, order_depth: OrderDepth, depth: int) -> float:
        # (Same implementation as V3)
        sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)
        sorted_asks = sorted(order_depth.sell_orders.items())
        total_bid_vol = sum(vol for price, vol in sorted_bids[:depth])
        total_ask_vol = sum(abs(vol) for price, vol in sorted_asks[:depth])
        denominator = total_bid_vol + total_ask_vol
        if denominator == 0: return 0.0
        return (total_bid_vol - total_ask_vol) / denominator

    def update_ema(self, current_value: float, product: str, state_dict: Dict[str, Any], alpha: float):
         # (Same implementation as V3)
        state_dict.setdefault("ema_prices", {})
        if state_dict["ema_prices"].get(product) is None:
            state_dict["ema_prices"][product] = current_value
        else:
            state_dict["ema_prices"][product] = alpha * current_value + (1 - alpha) * state_dict["ema_prices"][product]

    def update_history_and_stats(self, product: str, current_wmp: float | None, state_dict: Dict[str, Any]):
         # (Same implementation as V3)
        state_dict.setdefault("price_history", {}).setdefault(product, [])
        state_dict.setdefault("std_devs", {}).setdefault(product, 0.0)
        state_dict.setdefault("last_wmp", {}).setdefault(product, None)

        last_std_dev = state_dict["std_devs"].get(product, 0.0)

        if current_wmp is not None:
            history = state_dict["price_history"][product]
            history.append(current_wmp)
            max_len = PARAMS["shared"]["max_history_length"]
            if len(history) > max_len:
                state_dict["price_history"][product] = history[-max_len:]

            if len(state_dict["price_history"][product]) >= 5:
                try:
                    state_dict["std_devs"][product] = statistics.stdev(state_dict["price_history"][product])
                except statistics.StatisticsError:
                     state_dict["std_devs"][product] = 0.0
            else:
                 state_dict["std_devs"][product] = last_std_dev

            state_dict["last_wmp"][product] = current_wmp
        else:
             state_dict["std_devs"][product] = last_std_dev


    def calculate_fair_value(self, product: str, wmp: float | None, order_depth: OrderDepth, state_dict: Dict[str, Any]) -> float | None:
         # (Same implementation as V3)
        params = PARAMS[product]
        state_dict.setdefault("ema_prices", {}).setdefault(product, None)
        state_dict.setdefault("fair_values", {}).setdefault(product, None)
        state_dict.setdefault("last_wmp", {}).setdefault(product, None)
        last_ema = state_dict["ema_prices"][product]
        last_fv = state_dict["fair_values"][product]
        last_wmp_val = state_dict["last_wmp"][product]
        current_price_point = wmp
        if current_price_point is None: current_price_point = last_wmp_val
        if current_price_point is None: current_price_point = last_ema
        if current_price_point is None: current_price_point = last_fv
        if current_price_point is None and product == 'RAINFOREST_RESIN':
            current_price_point = params.get("fair_value_anchor")
            if current_price_point is None:
                 logger.print(f"CRITICAL WARNING: Anchor FV not defined for {product}") # Use logger
                 return None
        elif current_price_point is None:
             logger.print(f"CRITICAL WARNING: Cannot determine price point for {product}. Skipping FV calc.") # Use logger
             return None

        fair_value = None
        if product == 'RAINFOREST_RESIN':
             anchor = params["fair_value_anchor"]
             alpha = params["anchor_blend_alpha"]
             fair_value = alpha * current_price_point + (1 - alpha) * anchor
        elif product == 'KELP':
            self.update_ema(current_price_point, product, state_dict, params["ema_alpha"])
            fair_value = state_dict["ema_prices"][product]
            imbalance = self.calculate_order_book_imbalance(order_depth, params["imbalance_depth"])
            std_dev = state_dict["std_devs"].get(product, 0.0)
            safe_std_dev = max(std_dev, 0.1)
            spread_guess = max(params["min_spread"], round(safe_std_dev * params["volatility_spread_factor"]))
            spread_guess = min(spread_guess, 10)
            adj_factor = params["imbalance_fv_adjustment_factor"]
            fv_adjustment = imbalance * adj_factor * spread_guess
            fair_value += fv_adjustment
        else:
            fair_value = current_price_point

        state_dict["fair_values"][product] = fair_value
        return fair_value


    def manage_orders(self, product: str, position: int, limit: int, fair_value: float, std_dev: float, order_depth: OrderDepth, state_dict: Dict[str, Any]) -> List[Order]:
        """Generates orders using product-specific logic and parameters."""
        # (This is the core trading logic from V3 - **replace `print` with `logger.print`**)
        params = PARAMS[product]
        orders: List[Order] = []
        buy_room = limit - position
        sell_room = limit + position

        current_base_qty = params["base_order_qty"]
        if product == 'KELP':
             volatility = std_dev
             max_vol = params["max_volatility_for_qty_reduction"]
             min_qty_factor = params["min_volatility_qty_factor"]
             if max_vol > 0 and volatility > 0:
                  qty_factor = max(min_qty_factor, 1.0 - (volatility / max_vol) * (1.0 - min_qty_factor) )
                  current_base_qty = max(1, round(params["base_order_qty"] * qty_factor))

        take_profit_threshold = PARAMS["shared"]["take_profit_threshold"]
        orders_to_add: List[Order] = []

        if order_depth.sell_orders and buy_room > 0:
            best_ask = min(order_depth.sell_orders.keys())
            if best_ask < fair_value - take_profit_threshold:
                best_ask_vol = abs(order_depth.sell_orders[best_ask])
                qty_to_buy = min(best_ask_vol, buy_room)
                if qty_to_buy > 0:
                    logger.print(f"SNIPE BUY {product}: {qty_to_buy}x{best_ask} (FV: {fair_value:.2f})") # Use logger
                    orders_to_add.append(Order(product, best_ask, qty_to_buy))
                    buy_room -= qty_to_buy

        if order_depth.buy_orders and sell_room > 0:
             best_bid = max(order_depth.buy_orders.keys())
             if best_bid > fair_value + take_profit_threshold:
                best_bid_vol = order_depth.buy_orders[best_bid]
                qty_to_sell = min(best_bid_vol, sell_room)
                if qty_to_sell > 0:
                    logger.print(f"SNIPE SELL {product}: {qty_to_sell}x{best_bid} (FV: {fair_value:.2f})") # Use logger
                    orders_to_add.append(Order(product, best_bid, -qty_to_sell))
                    sell_room -= qty_to_sell

        orders.extend(orders_to_add)

        reversion_buy_price = None
        reversion_sell_price = None
        if product == 'RAINFOREST_RESIN':
             anchor = params['fair_value_anchor']
             reversion_threshold = params.get('reversion_threshold', 2)

             if order_depth.sell_orders and buy_room > 0:
                  best_ask = min(order_depth.sell_orders.keys())
                  if best_ask < anchor - reversion_threshold and not any(o.price == best_ask and o.quantity > 0 for o in orders):
                       qty_to_buy = min(abs(order_depth.sell_orders[best_ask]), buy_room)
                       if qty_to_buy > 0:
                            logger.print(f"MEAN REVERT BUY {product}: {qty_to_buy}x{best_ask} (Anchor: {anchor})") # Use logger
                            orders.append(Order(product, best_ask, qty_to_buy))
                            buy_room -= qty_to_buy
                            reversion_buy_price = best_ask

             if order_depth.buy_orders and sell_room > 0:
                  best_bid = max(order_depth.buy_orders.keys())
                  if best_bid > anchor + reversion_threshold and not any(o.price == best_bid and o.quantity < 0 for o in orders):
                       qty_to_sell = min(order_depth.buy_orders[best_bid], sell_room)
                       if qty_to_sell > 0:
                            logger.print(f"MEAN REVERT SELL {product}: {qty_to_sell}x{best_bid} (Anchor: {anchor})") # Use logger
                            orders.append(Order(product, best_bid, -qty_to_sell))
                            sell_room -= qty_to_sell
                            reversion_sell_price = best_bid

        min_spread = params["min_spread"]
        safe_std_dev = max(std_dev, 0.1)
        vol_spread = round(safe_std_dev * params["volatility_spread_factor"])
        dynamic_spread = max(min_spread, vol_spread)
        dynamic_spread = min(dynamic_spread, 10)
        inventory_skew = round(position * params["inventory_skew_factor"])
        target_bid = fair_value - (dynamic_spread / 2.0) - inventory_skew
        target_ask = fair_value + (dynamic_spread / 2.0) - inventory_skew
        target_ask = max(target_ask, target_bid + min_spread)
        bid_price = math.floor(target_bid)
        ask_price = math.ceil(target_ask)

        can_place_mm_bid = not (product == 'RAINFOREST_RESIN' and reversion_buy_price is not None and bid_price >= reversion_buy_price)
        can_place_mm_ask = not (product == 'RAINFOREST_RESIN' and reversion_sell_price is not None and ask_price <= reversion_sell_price)

        if buy_room > 0 and can_place_mm_bid:
            if not order_depth.sell_orders or bid_price < min(order_depth.sell_orders.keys()):
                 bid_qty = min(current_base_qty, buy_room)
                 if not any(o.price == bid_price and o.quantity > 0 for o in orders):
                     logger.print(f"MM BID {product}: {bid_qty}x{bid_price} (Sprd:{dynamic_spread}, Skew:{inventory_skew:.1f}, Qty:{current_base_qty})") # Use logger
                     orders.append(Order(product, bid_price, bid_qty))

        if sell_room > 0 and can_place_mm_ask:
             if not order_depth.buy_orders or ask_price > max(order_depth.buy_orders.keys()):
                 ask_qty = min(current_base_qty, sell_room)
                 if not any(o.price == ask_price and o.quantity < 0 for o in orders):
                      logger.print(f"MM ASK {product}: {ask_qty}x{ask_price} (Sprd:{dynamic_spread}, Skew:{inventory_skew:.1f}, Qty:{current_base_qty})") # Use logger
                      orders.append(Order(product, ask_price, -ask_qty))

        return orders

    def handle_conversions(self, product: str, position: int, observation: ConversionObservation) -> int:
         # (Same implementation as V3 - returns 0 for tutorial)
         return 0

    # --- Main Run Method ---
    def run(self, state: TradingState) -> Tuple[Dict[Symbol, List[Order]], int, str]:
        """
        Main trading logic method called each iteration. Now uses logger.
        """
        # Initialize final result variables
        result: Dict[Symbol, List[Order]] = {}
        total_conversions = 0
        traderData = "" # Default empty trader data

        try: # Wrap main logic in try-except to ensure flush always runs
            state_dict = self.deserialize_state(state.traderData)

            for product in POSITION_LIMITS:
                if product not in state.order_depths:
                    result[product] = []
                    continue

                order_depth = state.order_depths[product]
                current_position = state.position.get(product, 0)
                limit = POSITION_LIMITS[product]

                wmp = self.calculate_weighted_mid_price(order_depth)
                self.update_history_and_stats(product, wmp, state_dict)
                std_dev = state_dict["std_devs"].get(product, 0.0)

                fair_value = self.calculate_fair_value(product, wmp, order_depth, state_dict)

                if fair_value is None:
                     logger.print(f"Skipping {product} at ts {state.timestamp} - FV could not be determined.")
                     result[product] = []
                     continue

                # Log status using logger.print instead of standard print
                wmp_str = f'{wmp:.2f}' if wmp is not None else 'N/A'
                logger.print(f"[{product}] Pos:{current_position:>3}/{limit:<3} | WMP:{wmp_str:>8} | FV:{fair_value:>8.2f} | SD:{std_dev:>6.2f}")

                product_orders = self.manage_orders(product, current_position, limit, fair_value, std_dev, order_depth, state_dict)
                result[product] = product_orders

            traderData = self.serialize_state(state_dict) # Serialize final state

        except Exception as e:
            # Log any unexpected exceptions during the run
            logger.print(f"ERROR in Trader.run: {e}")
            # Optionally re-raise or handle differently
            # result, total_conversions, traderData might be in intermediate states
            # Ensure defaults or last known good values are used for flushing
            result = result if isinstance(result, dict) else {}
            total_conversions = total_conversions if isinstance(total_conversions, int) else 0
            traderData = traderData if isinstance(traderData, str) else ""


        # **CRITICAL:** Call logger.flush at the end, before returning
        # Pass the final computed result, conversions, and traderData
        logger.flush(state, result, total_conversions, traderData)

        # Return the same values that were passed to flush
        return result, total_conversions, traderData