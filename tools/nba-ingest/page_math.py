# Vale, Tue 10/13/2026. THE PROPS PAGE'S ARITHMETIC in one place, for the page to equal to four decimals (order PROPS-ORDER.md).
#   python3 props/page_math.py                      prints the worked example of the order
import math
K={"points":3.39,"rebounds":11.39,"assists":15.86,"threes":8.18,"pts+reb+ast":5.27,"pts+ast":4.72,"pts+reb":4.72,"ast+reb":11.39,"steals":8.18,"blocks":6.57,"stl+blk":9.14}
KM={"points":11.24,"rebounds":28.46,"assists":24.85,"threes":14.74,"steals":9.64,"blocks":7.2,"pts+reb+ast":18.89,"pts+ast":15.53,"pts+reb":15.37,"ast+reb":29.09,"stl+blk":13.22}
# KM, THE MARKET'S LAW (Vale 10/14): the spread of a player around his own mean, game to game (development months, players with 20+ games
# of 20+ minutes). It turns the market's line (its median) into the market's number (its mean). K stays the spread around OUR number,
# for our own chances. With K the market's points number ran +1.48 high on the 2026 playoffs; with KM +0.52 (props/k_points_test.py).
W_OURS=0.40      # the with-props number: 0.4 x ours + 0.6 x the market's
C_BELIEVE=0.10   # the chance used for AdjEV: the market's, moved this far toward ours in log-odds (EV itself uses our own chance)
def _pmf(x,mu,k): return math.exp(math.lgamma(x+k)-math.lgamma(k)-math.lgamma(x+1)+k*math.log(k/(k+mu))+x*math.log(mu/(k+mu)))
def chances(line,mu,k):
    """(chance of the over, of the under, of a push) for a count with mean mu and variance mu + mu^2/k"""
    top=int(math.floor(line)); c=sum(_pmf(x,mu,k) for x in range(0,top+1)); eq=_pmf(top,mu,k) if float(line)==top else 0.0; return max(1e-9,1-c),max(1e-9,c-eq),eq
def implied(american): o=float(str(american).replace("+","")); return (-o)/((-o)+100) if o<0 else 100/(o+100)
def pays(american): o=float(str(american).replace("+","")); return 100/(-o) if o<0 else o/100
def market_over(price_over,price_under): a,b=implied(price_over),implied(price_under); return a/(a+b)      # the margin taken out
def market_number(line,p_over,k):
    lo,hi=0.01,max(80.0,line*4)
    for _ in range(60):
        mid=(lo+hi)/2; so,su,_=chances(line,mid,k)
        if so/(so+su)<p_over: lo=mid
        else: hi=mid
    return (lo+hi)/2
def chance_at_least(v,mu,k):
    """P(X >= v) for a count with mean mu and size k (a milestone "v+")"""
    v=int(math.ceil(v)); return max(1e-9,1-sum(_pmf(x,mu,k) for x in range(0,v)))
MILESTONE={"Points Milestones":"points","Rebounds Milestones":"rebounds","Assists Milestones":"assists","3-Point Field Goals Milestones":"threes",
           "Points + Assists + Rebounds Milestones":"pts+reb+ast","Points + Assists Milestones":"pts+ast","Points + Rebounds Milestones":"pts+reb"}
def milestone_row(market,threshold,price,ours,markets_number=None):
    """A one-price market "threshold+" (Vale 10/14). EV: our chance of threshold or more, from our number, against the price as it is.
    AdjEV: the market's chance of it, from the market's number fitted on the same stat's main two-price line (so its margin is already
    out), moved C_BELIEVE toward ours in log-odds. With no main line for that stat there is no market's number: AdjEV is empty, never
    guessed from the one price (that price carries the book's margin and nothing to take it out with)."""
    k=K[market]; p=chance_at_least(threshold,ours,k); pay=pays(price); ev=p*pay-(1-p)
    out=dict(market=market,threshold=threshold,price=price,ours=ours,ourChance=round(p,4),ev=round(ev,4),marketsNumber=None,marketChance=None,adjChance=None,adjEv=None)
    if markets_number:
        pm=chance_at_least(threshold,markets_number,KM[market]); lg=lambda q:math.log(q/(1-q)); pa=1/(1+math.exp(-(lg(min(pm,1-1e-9))+C_BELIEVE*(lg(min(p,1-1e-9))-lg(min(pm,1-1e-9))))))
        out.update(marketsNumber=round(markets_number,4),marketChance=round(pm,4),adjChance=round(pa,4),adjEv=round(pa*pay-(1-pa),4))
    return out
def row(market,line,price_over,price_under,ours):
    k=K[market]; pm=market_over(price_over,price_under); mk=market_number(line,pm,KM[market]); withp=W_OURS*ours+(1-W_OURS)*mk
    so,su,_=chances(line,ours,k); po_model=so/(so+su); lg=lambda p:math.log(p/(1-p)); p=1/(1+math.exp(-(lg(pm)+C_BELIEVE*(lg(po_model)-lg(pm)))))
    # THE PUSH on a whole-number line, for AdjEV (Vale 10/14, on Mason's finding): from the same two laws AdjEV is made of, the market's
    # (its number, KM) moved C_BELIEVE toward ours (our number, K). Never from the with-props number: W_OURS belongs to that display
    # number only and stays out of both EV columns. EV's own push is ours, inside our chances above.
    _,_,push_m=chances(line,mk,KM[market]); _,_,push_o=chances(line,ours,k); push=(1-C_BELIEVE)*push_m+C_BELIEVE*push_o; po=p*(1-push); pu=(1-p)*(1-push)
    # Chris 10/13: "show AdjEV and EV. the number is the number." EV is our own number's chance, nothing done to it (so, su and the
    # push all from OUR number). AdjEV is the market's chance moved C_BELIEVE of the way toward ours.
    return dict(market=market,line=line,ours=ours,marketsNumber=round(mk,4),withProps=round(withp,4),marketChanceOver=round(pm,4),ourChanceOver=round(so,4),ourChanceUnder=round(su,4),
                evOver=round(so*pays(price_over)-su,4),evUnder=round(su*pays(price_under)-so,4),
                adjChanceOver=round(po,4),adjChanceUnder=round(pu,4),chancePush=round(push,4),adjEvOver=round(po*pays(price_over)-pu,4),adjEvUnder=round(pu*pays(price_under)-po,4))
if __name__=="__main__":
    for a in (("points",18.5,-112,-114,16.0),("points",18.5,-112,-114,24.0),("rebounds",7.0,+105,-130,8.2),("threes",2.5,-140,+110,1.6)): print(row(*a))
