import Utility as util

if __name__ == "__main__":
    path = util.get_path()
    img = util.load_img(path)
    #room = util.get_room(path)
    #util.quickPlot(img, f"Imported Image in {room.name}")
    util.quick_plot(img, f"Imported Image")
