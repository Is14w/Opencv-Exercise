import cv2 as cv
import Commons as cm

filepath = r"images/objects.jpg"

if __name__ == "__main__":
    # 读取图片
    img = cm.read_image(filepath)

    # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    # img_show(img)

    # 获取所有物体
    objects = cm.get_objects(img)

    # 获取目标图片
    target_img = cm.img_split(img, mode = "right", ratio = 0.85, fill = True)

    img_drawn = target_img.copy()

    # 对每个物体进行匹配
    for object in objects:
        loc = cm.find_target(object, target_img)

        x, y = loc
        h, w = object.shape[:2]

        cv.rectangle(img_drawn, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cm.img_show(img_drawn)