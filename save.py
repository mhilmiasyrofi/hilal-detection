# read external file
fh = open("parameters.txt", "r")
parameters = fh.readlines()
fh.close()
# remove new line string
parameters = [s.replace('\n', '') for s in parameters]

# get each variables
image_enhancement_mode = int(parameters[parameters.index("image_enhancement_mode")+1])
power = int(parameters[parameters.index("power")+1])
gamma = int(parameters[parameters.index("gamma")+1])
clip_limit = int(parameters[parameters.index("clip_limit")+1])
tile_grid_size = int(parameters[parameters.index("tile_grid_size")+1])
blur_size = int(parameters[parameters.index("blur_size")+1])
canny_min_val = int(parameters[parameters.index("canny_min_val")+1])
canny_max_val = int(parameters[parameters.index("canny_max_val")+1])
cht_min_dist = int(parameters[parameters.index("cht_min_dist")+1])
cht_param1 = int(parameters[parameters.index("cht_param1")+1])
cht_param2 = int(parameters[parameters.index("cht_param2")+1])
cht_min_radius = int(parameters[parameters.index("cht_min_radius")+1])
cht_max_radius = int(parameters[parameters.index("cht_max_radius")+1])

# update variable value
parameters[parameters.index("cht_max_radius")+1] = str(image_enhancement_mode)
parameters[parameters.index("power")+1] = str(power)
parameters[parameters.index("gamma")+1] = str(gamma)
parameters[parameters.index("clip_limit")+1] = str(clip_limit)
parameters[parameters.index("tile_grid_size")+1] = str(tile_grid_size)
parameters[parameters.index("blur_size")+1] = str(blur_size)
parameters[parameters.index("canny_min_val")+1] = str(canny_min_val)
parameters[parameters.index("canny_min_val")+1] = str(canny_min_val)
parameters[parameters.index("canny_max_val")+1] = str(canny_max_val)
parameters[parameters.index("cht_min_dist")+1] = str(cht_min_dist)
parameters[parameters.index("cht_param1")+1] = str(cht_param1)
parameters[parameters.index("cht_param2")+1] = str(cht_param2)
parameters[parameters.index("cht_min_radius")+1] = str(cht_min_radius)
parameters[parameters.index("cht_max_radius")+1] = str(cht_max_radius)


fw = open("parameters.txt", "w")

[fw.write(p + "\n") for p in parameters]

fw.close()
