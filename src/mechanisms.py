import numpy as np


def myerson_auction(bids, reserve_prices):
    """
    Run Myerson's optimal auction with personalized reserve prices
    
    Parameters:
    - bids: list of bids from each bidder
    - reserve_prices: reserve price for each bidder
    
    Returns:
    - allocation: who gets the item (index of winner, -1 if no winner)
    - payment: payment from each bidder
    """
    n = len(bids)
    allocation = -1  # No winner by default
    payments = [0] * n
    
    # Apply reserve prices
    adjusted_bids = [max(0, bids[i] - reserve_prices[i]) + reserve_prices[i] 
                    if bids[i] >= reserve_prices[i] else 0 
                    for i in range(n)]
    
    # Find winner (bidder with highest adjusted bid)
    if max(adjusted_bids) > 0:
        allocation = np.argmax(adjusted_bids)
        
        # Find second highest adjusted bid (excluding zero bids)
        nonzero_bids = [b for b in adjusted_bids if b > 0]
        if len(nonzero_bids) > 1:
            second_price = sorted(nonzero_bids)[-2]
        else:
            second_price = reserve_prices[allocation]
        
        payments[allocation] = second_price
    
    return allocation, payments


def vickrey_auction(bids):
    """
    Run Vickrey (second-price) auction
    
    Parameters:
    - bids: list of bids from each bidder
    
    Returns:
    - allocation: who gets the item (index of winner, -1 if no winner)
    - payment: payment from each bidder
    """
    n = len(bids)
    allocation = -1  # No winner by default
    payments = [0] * n
    
    if max(bids) > 0:
        allocation = np.argmax(bids)
        
        # Find second highest bid
        if len(bids) > 1:
            second_price = sorted(bids)[-2]
        else:
            second_price = 0
        
        payments[allocation] = second_price
    
    return allocation, payments


def anonymous_price_auction(bids, reserve_price):
    """
    Run anonymous price auction with a single reserve price
    
    Parameters:
    - bids: list of bids from each bidder
    - reserve_price: anonymous reserve price
    
    Returns:
    - allocation: who gets the item (index of winner, -1 if no winner)
    - payment: payment from each bidder
    """
    n = len(bids)
    allocation = -1  # No winner by default
    payments = [0] * n
    
    # Filter bids below reserve price
    qualified_bids = [(i, b) for i, b in enumerate(bids) if b >= reserve_price]
    
    if qualified_bids:
        # Find winner (bidder with highest bid above reserve)
        winner_idx, winner_bid = max(qualified_bids, key=lambda x: x[1])
        allocation = winner_idx
        
        # Find payment (max of second highest bid and reserve price)
        if len(qualified_bids) > 1:
            second_price = sorted([b for _, b in qualified_bids])[-2]
        else:
            second_price = reserve_price
        
        payments[allocation] = max(second_price, reserve_price)
    
    return allocation, payments