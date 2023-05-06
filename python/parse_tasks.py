import json

with open("data/tasks.json") as f:
    data = json.load(f)

print(data.keys())
print(data["name"])
print(len(data["lists"]))
# print(len(data["cards"]))
# print(data["cards"][0])
print(data[""])

lists = data["lists"]
for cur_list in lists:
    if cur_list["name"] == "Todo Python" or cur_list["name"] == "In-progress Python":
        # print(cur_list.keys(), cur_list["name"])
        a = cur_list
        # print(a)
