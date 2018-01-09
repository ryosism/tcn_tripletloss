from collections import OrderedDict
import json
import os

def make_one_pic():
    # print("anchor↓")
    anchor = input()
    if anchor == "":
        return 1
    anchor = os.path.relpath(anchor)

    # print("positive↓")
    positive = input()
    positive = os.path.relpath(positive)

    # ファイル名を取り出して
    filename = os.path.basename(positive)
    # ファイル名の拡張子と分離して、フレーム番号を手に入れる
    path, ext = os.path.splitext(os.path.basename(filename))
    index = int(path)

    dict = {'anchor': anchor, 'positive': positive, 'positive_index': index}
    return dict

def make_PIC():
    PIC = []

    while(1):
        pic = make_one_pic()
        if pic == 1:
            break

        PIC.append(pic)

    return PIC

def make_recipe():
    print("recipe_id")
    recipe_id = input()
    recipe = []
    recipe.append({"folder": recipe_id})
    recipe.append({"pic": make_PIC()})

    return recipe

def make_RECIPES():
    RECIPES = []
    while(1):
        recipe = make_recipe()
        if len(recipe[1]['pic']) == 0:
            break

        RECIPES.append(recipe)

    return RECIPES

if __name__ == '__main__':
    print("folder_num")
    num = input()
    print("料理名")
    name = input()

    dict = []
    dict.append({"name": name})
    dict.append(make_RECIPES())

    f = open("recipe{0}.json".format(num), "w")
    json.dump(dict, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))
