import pandas as pd
from datamodel import * # Import all classes from datamodel.py
from trader import Trader # Import your Trader class
import time # To measure execution time

# --- Configuration ---
# IMPORTANT: Update these filenames to match your downloaded sample data
ORDER_BOOK_CSV_PATH = 'data/Users/karthikkallam/Documents/imc_prosperity_algo/data/dc74c346-7f23-421a-890a-7f3006671568.csv'
# TRADE_CSV_PATH = 'data/YOUR_SAMPLE_TRADE_DATA.csv' # Optional, if needed for FV calc

# --- Helper Function to Parse Order Book Data ---
# IMPORTANT: This function MUST be adapted based on your CSV format.
#            Inspect your CSV columns carefully.
# Example assumption: CSV has columns like 'timestamp', 'product', 'bid_price_1', 'bid_volume_1', 'ask_price_1', 'ask_volume_1', etc.
def parse_order_depth_from_row(row, product: str) -> OrderDepth:
    depth = OrderDepth()
    # Example: Assuming up to 3 levels of depth provided
    for i in range(1, 4): # Adjust range based on CSV columns
        bid_price_col = f'bid_price_{i}'
        bid_vol_col = f'bid_volume_{i}'
        ask_price_col = f'ask_price_{i}'
        ask_vol_col = f'ask_volume_{i}'

        if bid_price_col in row and pd.notna(row[bid_price_col]) and row[bid_vol_col] > 0:
            depth.buy_orders[int(row[bid_price_col])] = int(row[bid_vol_col])
        if ask_price_col in row and pd.notna(row[ask_price_col]) and row[ask_vol_col] > 0:
            # Remember sell orders use negative quantity in datamodel
            depth.sell_orders[int(row[ask_price_col])] = -int(row[ask_vol_col]) 
            
    # Ensure bids are sorted high to low, asks low to high (optional, good practice)
    depth.buy_orders = dict(sorted(depth.buy_orders.items(), reverse=True))
    depth.sell_orders = dict(sorted(depth.sell_orders.items()))
    return depth

# --- Main Simulation Loop ---
def run_simulation():
    print(f"Loading data from {ORDER_BOOK_CSV_PATH}...")
    try:
        market_data_df = pd.read_csv(ORDER_BOOK_CSV_PATH, delim_whitespace=True) 
        print(f"Data loaded. Columns: {market_data_df.columns.tolist()}")
    except FileNotFoundError:
        print(f"ERROR: File not found at {ORDER_BOOK_CSV_PATH}")
        return
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        return

    # Get unique products and timestamps from the data
    products = market_data_df['product'].unique()
    timestamps = sorted(market_data_df['timestamp'].unique())
    
    print(f"Found Products: {products}")
    print(f"Simulating {len(timestamps)} timestamps...")

    # Initialize trader and state variables
    trader = Trader()
    traderData = "" # Initial empty state string
    current_positions: Dict[Product, Position] = {p: 0 for p in products} # Start with zero positions

    # --- Loop through each timestamp in the sample data ---
    for i, ts in enumerate(timestamps):
        start_time = time.time()
        print(f"\n--- Timestamp: {ts} ---")
        
        # Filter data for the current timestamp
        ts_data = market_data_df[market_data_df['timestamp'] == ts]

        # Construct the TradingState for this timestamp
        listings: Dict[Symbol, Listing] = {}
        order_depths: Dict[Symbol, OrderDepth] = {}
        
        for product in products:
             product_row = ts_data[ts_data['product'] == product]
             if not product_row.empty:
                 row = product_row.iloc[0] # Get the first row for this product/timestamp
                 listings[product] = Listing(symbol=product, product=product, denomination="SEASHELLS")
                 order_depths[product] = parse_order_depth_from_row(row, product)
             else:
                  # Handle cases where a product might not have data at a specific timestamp if needed
                  listings[product] = Listing(symbol=product, product=product, denomination="SEASHELLS")
                  order_depths[product] = OrderDepth() # Empty book


        # TODO: Populate market_trades and own_trades if your strategy uses them
        #       This would likely involve processing the separate trade CSV or
        #       simulating fills based on orders from the *previous* iteration.
        #       For simplicity, starting with empty trade lists.
        own_trades = {p: [] for p in products}
        market_trades = {p: [] for p in products}

        # TODO: Populate observations if needed for specific strategies
        #       This might require additional columns in your sample data or assumptions
        observations = Observation(plainValueObservations={}, conversionObservations={})

        # Create the state object
        state = TradingState(
            traderData=traderData,
            timestamp=ts,
            listings=listings,
            order_depths=order_depths,
            own_trades=own_trades,
            market_trades=market_trades,
            position=current_positions.copy(), # Pass a copy
            observations=observations
        )

        # --- Call the Trader's run method ---
        try:
            result, conversions, traderData = trader.run(state)
            # Check execution time
            end_time = time.time()
            duration = (end_time - start_time) * 1000 # ms
            print(f"Trader.run() executed in {duration:.2f} ms")
            if duration > 900:
                print("WARNING: Execution time exceeded 900ms!")

            # --- Process Results (Simplified) ---
            # In a real backtest, you would simulate order fills here based on
            # the 'result' orders and the 'order_depths' of the *next* timestamp
            # and update 'current_positions' and calculate PnL.
            # For now, just print the intended orders.
            print(f"Intended Orders: {result}")
            print(f"Intended Conversions: {conversions}")

            # Basic position update (assuming orders fill immediately for simplicity - NOT ACCURATE FOR MM)
            # A proper backtest needs careful fill simulation!
            for product, orders in result.items():
                 for order in orders:
                      # This is a VERY rough approximation, only counts intended change
                      current_positions[product] = current_positions.get(product, 0) + order.quantity
                      
            print(f"Approx Updated Positions (Ignoring Fills): {current_positions}")


        except Exception as e:
            print(f"ERROR during trader.run() at timestamp {ts}: {e}")
            # Optionally break the loop or handle error recovery
            break # Stop simulation on error

    print("\n--- Simulation Finished ---")

# --- Run the main simulation function ---
if __name__ == "__main__":
    run_simulation()