# Traffic Sign Recognition Project (GTSRB)

### Intro

This is a project for traffic-sign recognition. In it, we built a deep learning model to look at images for traffic signs (like "Stop", "Speed limit 30", etc.) and tell you what they are. We used the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset for this. 

Basically, the goal was to get really higha accuracy without making the model super slow.

### What We Did

We focused on Transfer Learning because training from scratch takes forever.

1. **First Step:** We trained a **MobileNetV2** model. We picked this one because it's supposed to be lightweight and fast.

2. **Second Step:** We wanted to see if a bigger model was actually better, so we compared it against **ResNet50**.

### MobileNetV2 vs. ResNet50

We ran some experiments to see the tradeoff between the two models. Everyone says ResNet is powerful, but MobileNet is way faster with only a slight fall in accuracy.

Here is the comparison chart:

![](D:\ML\Traffic%20Project\tradeoff.png)

**Findings:**

- **MobileNetV2** is super light and fast. It worked reall well for what we needed.

- **ResNet50** is a beast but very heavy and slow.

- For this specific task, the accuracy difference wasn't huge enough to justify the extra time and space of ResNet, so we stuck with MobileNetV2 for our final app.

### Results

- **Test Accuracy:** We hit around **93-98%** (depending on the run).

- **App:** We built a little interactive web app using **Gradio**.

### How to Run It

If you want to try this on your own machine:

1. Clone this repo.

2. Make sure you have the libraries installed (TensorFlow, NumPy, Pandas, Matplotlib, Gradio, etc.).

3. Open `Gtsrb.ipynb` in Jupyter Notebook or Google Colab (preferably, if you don't want to waste days running this).

4. Run all the cells
