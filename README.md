# Articulate-Visions

## About
___
Visualizing Text to Image tools

This is the git repository that aims to combine both the API backend and the vercel hosted frontend for the website
https://www.articulatevisions.net/.

The goal for this project is to create a playground where users can learn how text-to-image diffusion models work by allowing
users to change various variables that will affect the output of the image.

The frontend code is written for the Next.js framework and hosted on vercel. 
The backend is built on the flask framework. 
The model is was made using the GLIDE model created by OpenAI in 2022 and detailed in [this](https://arxiv.org/abs/2112.10741v3) paper. 
And the model is a modified version of [this](https://github.com/openai/glide-text2im) code. 

## Startup
___

This project can be started numerous ways. The two easiest options are using linux_start.sh or python_startup_script.py.

first clone this repo and open a terminal in the Articulate-Visions directory.
Then you will need npm and python installed in order to run this project

### Using linux_start.sh (Linux and macOS only)
```bash
sh linux_start.sh
```

### Using python. (Must have python installed)
```bash
python3 python_startup_script.py
```

### Compiling and running yourself

You can start the backend and frontend up in either order. It is recommended to start the backend first, however.

Start backend (in python):
```python
from backend import app
app.main()
```
Start Frontend (in terminal)
```bash
npm install
npm run build
npm run start
```

## Credits
___

This project was made by [Jack Welsh](https://www.linkedin.com/in/jack-welsh-bb849b250/), [Cooper Brown](https://www.linkedin.com/in/cbrown987/), and [Conrad Ernst](https://www.linkedin.com/in/conradernst/)
