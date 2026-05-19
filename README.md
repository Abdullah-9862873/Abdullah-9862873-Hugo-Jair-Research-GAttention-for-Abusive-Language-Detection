# GAttention for Abusive Language Detection: Multimodal Hate Speech Detection on Memes

## What This Project Is About

This research focuses on detecting hate speech in memes - those images with text overlays that you see everywhere on social media. The challenge here is that a meme's meaning isn't just in the image or the text alone, but in how both modalities work together. A meme that looks completely innocent in isolation might carry a completely different meaning when you consider the image-text combination.

I built a system that can look at both the text and the image together to determine whether a meme is hateful or not. This is different from traditional text-only hate speech detection because memes often rely on visual context, sarcasm, and cultural references that text alone can't capture.

## What Triggered This Research

This work came about after studying Professor Hugo Jair Escalante's GAttention research (published at EMNLP 2025). The GAttention paper presented a novel way to think about attention mechanisms in multimodal models - specifically how to make the model explainable by looking at which parts of the input it's paying attention to when making a decision.

I wanted to explore whether CLIP's joint embedding space - which allows text and images to be compared in the same mathematical space - could be leveraged for hate speech detection. The question was: can we detect hate speech better by combining both modalities rather than using either one alone?

## What I Did

I built and compared several approaches:

1. **CLIP-based classifier**: I used CLIP's joint text-image embeddings and trained a classifier on top of them to detect hate speech.

2. **RoBERTa text-only**: I used a strong text-only baseline with RoBERTa to see how much performance we'd lose without visual context.

3. **Image-only classifier**: To understand the value of each modality separately, I also trained a classifier using only images.

4. **MAGA (Multimodal Attention Gated Architecture)**: My custom model that learns when to trust text vs. image predictions using a learned gating mechanism. It looks at both modalities and decides which one to weight more for each specific sample.

I trained on a dataset of 2,400 memes, validated on 300, and tested on 300. The dataset was balanced with roughly 50% hateful and 50% non-hateful content.

## Results I Got

Here's how the models performed on the test set:

| Model | Test F1 Score | Test Accuracy |
|-------|---------------|---------------|
| CLIP | 0.639 | 64.0% |
| RoBERTa (text-only) | 0.626 | 63.0% |
| MAGA (my model) | 0.636 | 63.7% |
| Image-only | 0.333 | 50.0% |

The results show that:
- **CLIP performed best** with an F1 score of 0.639, slightly outperforming text-only RoBERTa
- **Adding images helps** - CLIP outperformed the text-only model, showing that visual context adds value
- **Image-only performs poorly** - Using images alone gives essentially random predictions (50% accuracy), confirming that hate speech in memes relies heavily on text
- **MAGA didn't improve** over CLIP - The gating mechanism didn't provide additional benefit in this dataset

I also did a failure analysis and found that the model struggles with:
- 62 false positives (non-hateful memes incorrectly flagged as hateful)
- 46 false negatives (hateful memes that slipped through)

## Why This Matters in Daily Life

This research has several practical applications:

**Social Media Moderation**: Platforms like Twitter, Facebook, and Instagram could use this technology to automatically flag and review memes that may contain hate speech before they spread. This could help content moderators handle the overwhelming volume of user-generated content.

**Content Filtering for Parents**: Parents could use such systems to filter out potentially harmful content from what their children see online.

**Research Tool for Social Scientists**: Researchers studying the spread of hate speech online could use these models to analyze large datasets and understand patterns in how hate content evolves.

**Accessibility**: These models could help build more inclusive content recommendation systems that avoid promoting divisive or hateful content.

## Project Structure

- `data/` - Contains the dataset (train.json, val.json, test.json) with meme images and text
- `notebooks/` - Jupyter notebook with all the model training and evaluation code
- `results/` - Final results, visualizations, and failure analysis

## Dataset Details

The dataset contains memes collected from social media, each with:
- Text content (the text overlay on the meme)
- Image (the visual content)
- Binary label (hateful or not hateful)

The dataset is split into:
- Training: 2,400 samples (1,192 hateful, 1,208 non-hateful)
- Validation: 300 samples (158 hateful, 142 non-hateful)
- Test: 300 samples (150 hateful, 150 non-hateful)

## Future Work

This is a starting point. More work is needed to:
- Test on larger, more diverse datasets
- Improve the model's ability to understand sarcasm and cultural context
- Make the model more interpretable so we can understand why it makes certain predictions
- Reduce false positives to avoid silencing legitimate content