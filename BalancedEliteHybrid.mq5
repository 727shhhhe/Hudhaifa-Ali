//+------------------------------------------------------------------+
//|                                       BalancedEliteHybrid.mq5    |
//|          Balanced Elite Hybrid Strategy - XAUUSD (Gold) M1       |
//+------------------------------------------------------------------+
#property copyright "Hudhaifa"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

input int    mom_period        = 9;      // Momentum baseline period
input double body_expand_min   = 1.50;   // Candle expansion filter minimum (150%)
input double body_expand_max   = 4.00;   // Candle expansion filter maximum (400%)
input double base_lot          = 0.001;  // Nano-lot base size
input int    max_positions     = 2;      // Grid recovery cap
input double grid_step_points  = 12.0;   // Grid increment in points
input double tp_points         = 22.0;   // Take profit in points from basket average
input double trail_activate    = 12.00;  // Micro-trailing activation profit ($)
input double trail_retrace     = 4.00;   // Retracement from peak that closes basket ($)
input double hard_stop_cash    = 10.00;  // Hard stop loss per basket ($)
input ulong  magic_number      = 20240101;

CTrade  trade;
double  peak_profit = 0.0;
datetime last_bar_time = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(magic_number);
   trade.SetTypeFillingBySymbol(_Symbol);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
  }

//+------------------------------------------------------------------+
//| Basket helpers                                                   |
//+------------------------------------------------------------------+
int BasketCount()
  {
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)magic_number)
         continue;
      count++;
     }
   return(count);
  }

int BasketDirection()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)magic_number)
         continue;
      return((int)PositionGetInteger(POSITION_TYPE));
     }
   return(-1);
  }

double BasketAveragePrice()
  {
   double volume_sum = 0.0;
   double weighted   = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)magic_number)
         continue;
      double vol = PositionGetDouble(POSITION_VOLUME);
      weighted   += PositionGetDouble(POSITION_PRICE_OPEN) * vol;
      volume_sum += vol;
     }
   if(volume_sum <= 0.0)
      return(0.0);
   return(weighted / volume_sum);
  }

double BasketProfit()
  {
   double profit = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)magic_number)
         continue;
      profit += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
     }
   return(profit);
  }

double LastEntryPrice()
  {
   datetime newest = 0;
   double   price  = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)magic_number)
         continue;
      datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
      if(opened >= newest)
        {
         newest = opened;
         price  = PositionGetDouble(POSITION_PRICE_OPEN);
        }
     }
   return(price);
  }

void CloseBasket()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != (long)magic_number)
         continue;
      trade.PositionClose(ticket);
     }
   peak_profit = 0.0;
  }

//+------------------------------------------------------------------+
//| Signal generation                                                |
//+------------------------------------------------------------------+
bool Signal(bool &buy, bool &sell)
  {
   buy  = false;
   sell = false;

   int needed = mom_period + 3;
   MqlRates rates[];
   if(CopyRates(_Symbol, PERIOD_M1, 0, needed, rates) < needed)
      return(false);
   ArraySetAsSeries(rates, true);

   // Momentum baseline over a mom_period rolling window
   double mom      = rates[1].close - rates[1 + mom_period].close;
   double mom_prev = rates[2].close - rates[2 + mom_period].close;
   double mom_slope = mom - mom_prev;

   // Candle bodies
   double body      = MathAbs(rates[1].close - rates[1].open);
   double prev_body = MathAbs(rates[2].close - rates[2].open);
   if(prev_body <= 0.0)
      return(false);

   double ratio = body / prev_body;
   bool expansion = (ratio >= body_expand_min && ratio <= body_expand_max);

   bool is_green      = (rates[1].close > rates[1].open);
   bool prev_is_green = (rates[2].close > rates[2].open);
   bool opposite_dir  = (is_green != prev_is_green);

   if(mom > 0 && mom_slope >= 0 && expansion && opposite_dir)
      buy = true;

   if(mom < 0 && mom_slope <= 0 && expansion && opposite_dir)
      sell = true;

   return(buy || sell);
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double bid   = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask   = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   int count = BasketCount();

   //--- Risk & profit management on the open basket
   if(count > 0)
     {
      double profit = BasketProfit();

      // Strict hard stop loss
      if(profit <= -hard_stop_cash)
        {
         CloseBasket();
         return;
        }

      // Micro-trailing profit lock
      if(profit >= trail_activate)
        {
         if(profit > peak_profit)
            peak_profit = profit;
        }
      if(peak_profit >= trail_activate && (peak_profit - profit) >= trail_retrace)
        {
         CloseBasket();
         return;
        }

      // Take profit from basket average price
      int    dir = BasketDirection();
      double avg = BasketAveragePrice();
      if(avg > 0.0)
        {
         if(dir == POSITION_TYPE_BUY && bid >= avg + tp_points * point)
           {
            CloseBasket();
            return;
           }
         if(dir == POSITION_TYPE_SELL && ask <= avg - tp_points * point)
           {
            CloseBasket();
            return;
           }
        }

      // Grid recovery
      if(count < max_positions)
        {
         double last = LastEntryPrice();
         if(dir == POSITION_TYPE_BUY && ask <= last - grid_step_points * point)
            trade.Buy(base_lot, _Symbol, 0.0, 0.0, 0.0);
         if(dir == POSITION_TYPE_SELL && bid >= last + grid_step_points * point)
            trade.Sell(base_lot, _Symbol, 0.0, 0.0, 0.0);
        }
      return;
     }

   peak_profit = 0.0;

   //--- New entries evaluated once per closed bar
   datetime bar_time = (datetime)SeriesInfoInteger(_Symbol, PERIOD_M1, SERIES_LASTBAR_DATE);
   if(bar_time == last_bar_time)
      return;
   last_bar_time = bar_time;

   bool buy, sell;
   if(!Signal(buy, sell))
      return;

   if(buy)
      trade.Buy(base_lot, _Symbol, 0.0, 0.0, 0.0);
   else
      if(sell)
         trade.Sell(base_lot, _Symbol, 0.0, 0.0, 0.0);
  }
//+------------------------------------------------------------------+
