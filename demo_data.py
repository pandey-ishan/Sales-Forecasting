"""Synthetic Superstore-style data so the dashboard can be tried without a file.

THIS IS NOT REAL DATA. It mimics the public Superstore schema (including dd/mm/yyyy dates,
a postal code with a lost leading zero and one city with no postal code) so every part of the
app can be exercised. The app labels it clearly whenever it is loaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CITIES = [  # city, state, postal code, region
    ("New York City", "New York", 10024, "East"), ("Los Angeles", "California", 90036, "West"),
    ("San Francisco", "California", 94109, "West"), ("Seattle", "Washington", 98103, "West"),
    ("Houston", "Texas", 77095, "Central"), ("Chicago", "Illinois", 60610, "Central"),
    ("Philadelphia", "Pennsylvania", 19140, "East"), ("Henderson", "Kentucky", 42420, "South"),
    ("Fort Lauderdale", "Florida", 33311, "South"), ("Concord", "North Carolina", 28027, "South"),
    ("Jacksonville", "Florida", 32216, "South"), ("Dallas", "Texas", 75217, "Central"),
    ("Columbus", "Ohio", 43229, "East"), ("Denver", "Colorado", 80219, "West"),
    ("Phoenix", "Arizona", 85023, "West"), ("Atlanta", "Georgia", 30318, "South"),
    ("Boston", "Massachusetts", 2122, "East"),            # leading zero lost, like a numeric CSV column
    ("Burlington", "Vermont", None, "East"),              # missing postal code
    ("Minneapolis", "Minnesota", 55407, "Central"), ("Detroit", "Michigan", 48205, "Central"),
    ("Portland", "Oregon", 97206, "West"),
]

# category -> sub-category -> (log-price mean, log-price sd, code)
CATALOG = {
    "Furniture": {"Bookcases": (5.4, .55, "BO"), "Chairs": (5.6, .6, "CH"),
                  "Furnishings": (3.4, .7, "FU"), "Tables": (5.7, .7, "TA")},
    "Office Supplies": {"Art": (2.4, .6, "AR"), "Appliances": (4.3, .9, "AP"), "Binders": (3.2, 1.0, "BI"),
                        "Envelopes": (3.8, .6, "EN"), "Fasteners": (1.9, .5, "FA"), "Labels": (2.6, .6, "LA"),
                        "Paper": (3.3, .8, "PA"), "Storage": (4.5, .8, "ST"), "Supplies": (3.5, .9, "SU")},
    "Technology": {"Accessories": (4.4, .8, "AC"), "Copiers": (7.2, .7, "CO"),
                   "Machines": (6.2, .9, "MA"), "Phones": (5.5, .7, "PH")},
}
MONTH_F = {1: .65, 2: .6, 3: 1.0, 4: .85, 5: .9, 6: .95, 7: .9, 8: .95, 9: 1.4, 10: 1.0, 11: 1.7, 12: 1.8}
DOW_F = [1.1, 1.0, 1.0, 1.05, 1.15, .75, .55]  # Mon..Sun
FIRST = ["Claire", "Darrin", "Sean", "Brosina", "Irene", "Harold", "Pete", "Alejandro", "Zuschuss", "Ken",
         "Sandra", "Emily", "Eric", "Tracy", "Matt", "Gene", "Ruben", "Katherine", "Maria", "Nora"]
LAST = ["Gute", "Van Huff", "O'Donnell", "Hoffman", "Maddox", "Pawlan", "Grace", "Carroll", "Donatelli",
        "Brennan", "Zirpoli", "Hwang", "Dillard", "Nguyen", "Stevenson", "Mehta", "Banks", "Foster"]
SHIP = {"Standard Class": (.60, 4, 7), "Second Class": (.20, 2, 5), "First Class": (.15, 1, 3), "Same Day": (.05, 0, 0)}


def make_demo(seed: int = 7, start: str = "2014-01-01", end: str = "2017-12-30") -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    products = []
    for cat, subs in CATALOG.items():
        for sub, (mu, sd, code) in subs.items():
            for i in range(10):
                products.append(dict(
                    cat=cat, sub=sub, price=float(np.exp(rng.normal(mu, sd * .6))),
                    pid=f"{cat[:3].upper()}-{code}-{10000000 + len(products)}",
                    name=f"Demo {sub} Model {i + 1}",
                ))

    customers = []
    for i in range(120):
        f, l = rng.choice(FIRST), rng.choice(LAST)
        city = CITIES[rng.integers(len(CITIES))]
        customers.append(dict(
            cid=f"{f[0]}{l[0]}-{12000 + i}", name=f"{f} {l}", city=city,
            seg=rng.choice(["Consumer", "Corporate", "Home Office"], p=[.52, .30, .18]),
        ))

    modes, probs = list(SHIP), np.array([v[0] for v in SHIP.values()])
    dates = pd.date_range(start, end, freq="D")
    rows, row_id, order_no = [], 1, 100000
    for d in dates:
        lam = 2.6 * MONTH_F[d.month] * DOW_F[d.dayofweek] * (1 + .12 * (d.year - dates[0].year))
        for _ in range(rng.poisson(lam)):
            order_no += 1
            cu = customers[rng.integers(len(customers))]
            mode = rng.choice(modes, p=probs)
            lo, hi = SHIP[mode][1], SHIP[mode][2]
            ship = d + pd.Timedelta(days=int(rng.integers(lo, hi + 1)))
            for _ in range(int(rng.integers(1, 5))):
                p = products[rng.integers(len(products))]
                qty = int(rng.integers(1, 6))
                sales = p["price"] * qty * float(np.exp(rng.normal(0, .15)))
                if rng.random() < .01:
                    sales *= 4  # occasional big order -> outliers for the z-score view
                city, state, zipc, region = cu["city"]
                rows.append((
                    row_id, f"CA-{d.year}-{order_no}", d.strftime("%d/%m/%Y"), ship.strftime("%d/%m/%Y"), mode,
                    cu["cid"], cu["name"], cu["seg"], "United States", city, state,
                    np.nan if zipc is None else float(zipc), region, p["pid"], p["cat"], p["sub"], p["name"],
                    round(sales, 4),
                ))
                row_id += 1

    cols = ["Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Customer Name",
            "Segment", "Country", "City", "State", "Postal Code", "Region", "Product ID", "Category",
            "Sub-Category", "Product Name", "Sales"]
    return pd.DataFrame(rows, columns=cols)
