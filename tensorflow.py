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
    # convert to lowercase
    lowercase = tf.strings.lower(input_data)
    # remove punctuation
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')
    return no_punctuation

def vectorize_headline(text, label, vectorize_layer1):
    text = tf.expand_dims(text, -1)
    return vectorize_layer1(text), [label]
