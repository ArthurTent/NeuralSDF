#  Neural SDF
This repository is based on Blacke Mori's NeuralSDF Tutorial

- https://www.youtube.com/watch?v=8pwXpfi-0bU
- https://www.shadertoy.com/view/wtVyWK
- https://drive.google.com/drive/folders/13-ks7iyLyI0vcS38xq1eeFdaMdfNlUC8

The neural_experiments.zip is a copy of the google drive files.

The goal of this repository is to offer a easy to use snippet without the need of jupyter notebook.

## Requirements

- python3.12 (might also work with lower version, but haven't tested)
- pipenv

## Steps:
- clone this repository
- change to the repository folder and run:
    - pipenv sync
    - pipenv shell
    - unzip the neural_experiments.zip

- run a python shell within the pipenv environment: 

```
from siren import *

loader = load("neural_experiments/bunny2.obj")

# Parameters: loader, neural_network_dimension, neural_network_dimension, i_dont_know, use_cuda)
# If you have an nvidia gpu with cuda support, use True, otherwise use False
# Avoid too complex dimensions. the more complex the neural network gets, the longer the training time
# and also the resulted shader will be very slow.
siren = train_siren(loader, 16, 4, 15, True)

# Parameters: siren_neural_network, i_really_dont_know, float_precision)
# Try to use a high precision in the beginning and then lower it to find the sweet spot
serialize_to_shaderamp(siren, "f", 3)

```

The Code above will create a new file named ```shaderamp.frag```


## Use the new shader in ShaderAmp
In order to use the new shader you need ShaderAmp compiled locally.
Navigate one folder up again and checkout ShaderAmp

### Requirements
- npm

```
git clone https://github.com/ArthurTent/ShaderAmp.git
cd ShaderAmp
npm install
npm run dev
```

Copy the shaderamp.frag to:

```
ShaderAmp/dist/shaders
```

You also need a 'shaderamp.frag.meta' file with the following content:

```
{
    "author": "blackle",
    "modifiedBy": "ENTER YOUR NAME HERE",
    "shaderName": "shaderamp.frag",
    "url": "https://www.shadertoy.com/view/wtVyWK",
    "license": "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License",
    "licenseURL": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
    "shaderSpeed": 0.8
}
```


Navigate in Chrome/Chromium to chrome://extensions/

If you haven't done yet, enable "Developer mode" on the right top corner.
Afterwards you can "Load unpacked" extension from the top left corner.

Select the "dist" folder within ShaderAmp directory.


Keep in mind: after adding a new shader to ShaderAmp, you have to reload the whole Plugin through the [chrome://extensions/](chrome://extensions/) page. 

Now you can test your new shader.
Navigate to a page with sound, click the ShaderAmp Icon and press "Switch to ShaderAmp"

A new browser tab will open with ShaderAmp visualization. Stay on that page and click the ShaderAmp Icon again.
Now you can open ShaderAmp options menu. Open it and select the new shaderamp.frag shader from the options menu.

Enjoy!
