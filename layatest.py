import laya_mlx as laya

agent = laya.load("aac6fef/laya-mlx")
result = agent.predict(
    "From: buyer@acme-trading.de — Subject: order for ACME GmbH.",
    {
        "department": {
            "type": "choice",
            "instructions": "What SKU should the order be placed for ",
            "criteria": ["ACME GmbH", "ACME Trading GmbH"],
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
    },
)
print(result["answers"])
