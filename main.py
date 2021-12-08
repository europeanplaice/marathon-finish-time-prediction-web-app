import argparse

import pandas as pd
import seaborn as sns

from finish_time_predictor import Decoder, Encoder, FinishTimePredictor
from utils import (makedataset, preprocess_rawdata)

sns.set(font="ricty diminished")


def main(param=None):

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--do_all_splits_eval", action="store_true")
    parser.add_argument("--train_data_path", default='boston2017-2018.csv')
    parser.add_argument("--elapsed_time")
    parser.add_argument("--elapsed_time_what_if")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--full_record")
    parser.add_argument("--graph_save_path", default="estimation.jpg")
    parser.add_argument("--encoder_model_path",
                        default='trained_model/encoder/encoder')
    parser.add_argument("--decoder_model_path",
                        default='trained_model/decoder/decoder')

    if param is not None:
        args = parser.parse_args(param)
    else:
        args = parser.parse_args()
    if args.elapsed_time is not None:
        args.do_predict = True
    encoder = Encoder(args)
    decoder = Decoder(args)
    finish_time_predictor = FinishTimePredictor(encoder, decoder)
    if args.do_train or args.do_eval:

        df = pd.read_csv(args.train_data_path)
        # save_feather(df)
        data = preprocess_rawdata(df)

        train_dataset, eval_dataset, eval_onebatch_dataset = \
            makedataset(data, args, True)
        if args.do_train:
            finish_time_predictor.train(
                train_dataset, eval_dataset, args)
    if args.do_eval:
        finish_time_predictor.load_weights(args)
        finish_time_predictor.validate(eval_onebatch_dataset, args)
    if args.do_predict:
        args.elapsed_time = args.elapsed_time[:-1]
        print(args.elapsed_time)
        # args.elapsed_time = args.elapsed_time.replace(";", ":")
        # args.elapsed_time = args.elapsed_time.replace(" ", "")
        elapsed_time_list = args.elapsed_time.split(",")
        print(elapsed_time_list)
        for i in range(len(elapsed_time_list)):
            print(elapsed_time_list[i])
            # if len(elapsed_time_list[i].split(":")) == 2:
            #     elapsed_time_list[i] = "0:" + elapsed_time_list[i]
            # elif len(elapsed_time_list[i].split(":")) == 1:
            #     elapsed_time_list[i] = "0:" + elapsed_time_list[i] + ":0"
            # elapsed_time_list[i] = elapsed_time_list[i].replace("::", ":")
        finish_time_predictor.load_weights(args)
        if args.elapsed_time_what_if is not None:
            predicted_time = finish_time_predictor.predict(
                elapsed_time_list,
                args,
                args.elapsed_time_what_if.split(","),
            )
        else:
            predicted_time = finish_time_predictor.predict(
                elapsed_time_list,
                args,
            )
        return predicted_time


if __name__ == '__main__':
    main()
