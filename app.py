import logging
import time
import argparse
import importlib
from antispoofing.src.topics import BaseDataCapture, Celery, Liveproof, \
    Faceextractor, Faceverification
# from main import BaseLitModel
import torch
import cv2

from dotenv import dotenv_values
from pathlib import Path

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)

config = dotenv_values(".env")
DEFAULT_TOPIC = config["DEFAULT_TOPIC"]
DATA_DIR = Path(__file__).resolve().parents[1]


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'antispoofing.src.topics'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--topic_class", type=str, default=DEFAULT_TOPIC)
    temp_args, _ = parser.parse_known_args()
    topic_class = _import_class(
        f"antispoofing.src.topics.{temp_args.topic_class}")
    topic_group = parser.add_argument_group("Topic Args")
    topic_class.add_to_argparse(topic_group)
    args = parser.parse_args()

    logging.basicConfig(filename=f'./logs/biometrics.{temp_args.topic_class}.log',
                        filemode='w', format='%(levelname)s on -> %(name)s at %(asctime)s \n\t %(message)s')

    # start = str(
    #     args.start_timestamp) if args.start_timestamp != 0 else "beginning"
    # end = str(args.end_timestamp) if args.end_timestamp != int(
    #     sys.maxsize) else "now"
    # LOGGER.info(f'The interval time is: {start} and {end}')
    return parser


def main():
    initial_time = time.time()
    parser = _setup_parser()
    args = parser.parse_args()
    topic_class = _import_class(f"antispoofing.src.topics.{args.topic_class}")
    data_event_topic = topic_class(args)
    # start_date = datetime.fromtimestamp(data_event_topic.start_timestamp)
    # end_date = datetime.fromtimestamp(data_event_topic.end_timestamp) if data_event_topic.end_timestamp != int(
    #     sys.maxsize) else "now"
    # LOGGER.info(
    #     f'The interval time is: {start_date} and {end_date}')
    print(" ")
    print("*"*30)
    print(data_event_topic.info())
    print("*"*30)
    print(" ")
    LOGGER.info(data_event_topic.info())
    try:
        data_event_topic.capture_and_process_events()
    finally:
        time_min = int((time.time()-initial_time)/60)
        LOGGER.info(f"Run time in minutes: {time_min}")
        data_event_topic.close()
    return


if __name__ == '__main__':
    #  bin/kafka-console-consumer.sh --bootstrap-server localhost:7002 --topic nltcs-socket-requests --from-beginning
    main()
