#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 15:10:12 2025

@author: diyacowlagi
"""
#from lecture
import string
import tensorflow as tf
import re

def standardization(input_data):
    """
    Standardize text input by lowercasing and removing punctuation.

    Paramters:
    input_data (tf.Tensor or str): the text to be standardized

    Returns:
    string tensor with all characters lowercased and punctuation removed
    """
    # convert to lowercase
    lowercase = tf.strings.lower(input_data)
    # remove punctuation
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')
    return no_punctuation

def vectorize_headline(text, label, vectorize_layer1):
    """
    Vectorize a headline and pair it with its corresponding label.

    Paramters:
    text (tf.Tensor or str): The headline to be vectorized
    label (tf.Tensor or int): The label associated 
    vectorize_layer1 (tf.keras.layers.TextVectorization): A fitted TextVectorization layer used to transform the headline into vectors

    Returns:
    string tensor with all characters lowercased and punctuation removed
    """
    text = tf.expand_dims(text, -1)
    return vectorize_layer1(text), [label]
