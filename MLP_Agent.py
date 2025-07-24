#!/usr/bin/env python3
"""
train_future_trajectory.py

Train a future trajectory predictor on CSV data.

Columns:
start_time,end_time,duration_s,agent_id,chunk_id,duration_s_actual,
latitude,longitude,category,stop_name

Usage:
    python train_future_trajectory.py --csv /path/to/data.csv [options]
"""
import os
import time
import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Force Theano backend so we never hit tensorflow.keras internals
os.environ['KERAS_BACKEND'] = 'theano'
import keras
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def build_sequence_data(df):
    """
    Take a sorted DataFrame and produce feature rows/events 0..n-2 plus
    labels = next-event duration & category.
    """
    X_records, y_dur, y_cat = [], [], []
    for agent, grp in df.groupby('agent_id'):
        grp = grp.sort_values('start_time').reset_index(drop=True)
        if len(grp) < 2:
            continue
        for i in range(len(grp) - 1):
            curr, nxt = grp.iloc[i], grp.iloc[i+1]
            X_records.append({
                'agent_id': curr['agent_id'],
                'stop_name': curr['stop_name'],
                'category': curr['category'],
                'duration_s': curr['duration_s'],
                'duration_s_actual': curr['duration_s_actual'],
                'latitude': curr['latitude'],
                'longitude': curr['longitude'],
            })
            y_dur.append(nxt['duration_s'])
            y_cat.append(nxt['category'])
    return pd.DataFrame(X_records), np.array(y_dur), np.array(y_cat)


def main():
    parser = argparse.ArgumentParser(description="Train future trajectory predictor.")
    parser.add_argument('--csv',       type=str, required=True, help='Path to CSV file')
    parser.add_argument('--epochs',    type=int, default=1)
    parser.add_argument('--batch_size',type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=32, help='Embedding size')
    parser.add_argument('--lr',        type=float, default=0.001)
    parser.add_argument('--out_dir',   type=str, default='./models')
    args = parser.parse_args()

    # Load and sequence the data
    df = pd.read_csv(args.csv, parse_dates=['start_time','end_time'])
    X_df, y_dur, y_cat_raw = build_sequence_data(df)

    # Encode categorical features
    le_agent = LabelEncoder().fit(X_df['agent_id'])
    le_stop  = LabelEncoder().fit(X_df['stop_name'])
    le_cat   = LabelEncoder().fit(pd.concat([X_df['category'], pd.Series(y_cat_raw)]))

    X_df['agent_idx'] = le_agent.transform(X_df['agent_id'])
    X_df['stop_idx']  = le_stop.transform(X_df['stop_name'])
    X_df['cat_idx']   = le_cat.transform(X_df['category'])
    y_cat = le_cat.transform(y_cat_raw)
    y_cat = to_categorical(y_cat, num_classes=len(le_cat.classes_))

    # Numeric feature matrix
    X_num = X_df[['duration_s','duration_s_actual','latitude','longitude']].values

    # Model inputs
    in_agent = Input(shape=(1,), name='agent_input')
    in_stop  = Input(shape=(1,), name='stop_input')
    in_cat   = Input(shape=(1,), name='cat_input')
    in_num   = Input(shape=(4,), name='num_input')

    # Embeddings
    emb_agent = Embedding(input_dim=len(le_agent.classes_), output_dim=args.embed_dim)(in_agent)
    emb_stop  = Embedding(input_dim=len(le_stop.classes_),  output_dim=args.embed_dim)(in_stop)
    emb_cat   = Embedding(input_dim=len(le_cat.classes_),   output_dim=args.embed_dim)(in_cat)

    flat = Flatten()
    u, s, c = flat(emb_agent), flat(emb_stop), flat(emb_cat)

    # Concatenate and MLP
    x = Concatenate()([u, s, c, in_num])
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    # Outputs
    out_dur = Dense(1, activation='linear', name='duration')(x)
    out_cat = Dense(len(le_cat.classes_), activation='softmax', name='category')(x)

    model = Model(inputs=[in_agent,in_stop,in_cat,in_num], outputs=[out_dur,out_cat])
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss={'duration':'mse', 'category':'categorical_crossentropy'},
        metrics={'duration':'mae', 'category':'accuracy'}
    )

    # Prepare arrays and train
    X_user = X_df['agent_idx'].values
    X_stop = X_df['stop_idx'].values
    X_cat  = X_df['cat_idx'].values
    
    model.summary()

    model.fit(
        [X_user, X_stop, X_cat, X_num],
        [y_dur, y_cat],
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=2
    )

    # Save final model
    os.makedirs(args.out_dir, exist_ok=True)
    model_file = os.path.join(args.out_dir, f"future_traj_{int(time.time())}.h5")
    model.save(model_file)
    print(f"Model saved to {model_file}")


if __name__ == '__main__':
    main()
