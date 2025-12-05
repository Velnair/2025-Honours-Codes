# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Metavision SDK Get Started.
"""


from metavision_core.event_io import EventsIterator
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIAction, UIKeyEvent

def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision SDK Get Started sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '-a', '--acc-time', dest='accumulation_time_us', type=int, default=10000,
        help='Duration of the time slice to store in the rolling event buffer at each tracking step')
    
    parser.add_argument(
        '-i', '--input-event-file', dest='event_file_path', default="",
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used. "
        "If it's a camera serial number, it will try to open that camera instead.")

    parser.add_argument(
        '-f', '--frequency', dest='freq', type=float, default=1,
        help='Frequency of Galvo, which will be the duration of served event slice, in seconds')

    parser.add_argument(
        '-N', '--datapoints', dest='N', type=int, default=1000,
        help='Number of datapoints to obtain before shutting down script')
    
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    # Events iterator on Camera or event file
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=int(1e6/args.freq))	# get frequency in microseconds 
    
    global_counter = 0      # This will track how many events we processed
    global_max_t = 0        # This will track the highest timestamp we processed
    filename = f'{(args.freq)}Hz-data'
    lim = args.N

    # Process events
    with open(filename, 'w') as f:                  # implement writing loops
        for i, evs in enumerate(mv_iterator):       # track the number of iteration & elements evs 

            if evs.size == 0:
                print("The current event buffer is empty.")
            else:
                min_t = evs['t'][0]   # Get the timestamp of the first event of this callback
                max_t = evs['t'][-1]  # Get the timestamp of the last event of this callback
                global_max_t = max_t  # Events are ordered by timestamp, so the current last event has the highest timestamp

                counter = evs.size         # Local counter
                global_counter += counter  # Increase global counter
                    
                print(f"Datapoints collected: {int(i/lim*100)}%",  end = '\r')	#recussive line update

                # write counter in a .txt file
                f.write(f'{counter} \n')

                # check if past or within limit to keep iteration going
                if i >= lim:
                    print(f'{lim} Datapoints obtained')
                    break

    # Print the global statistics
    duration_seconds = global_max_t / 1.0e6
    print(f"There were {global_counter} events in total.")
    print(f"The total duration was {duration_seconds:.2f} seconds.")
    if duration_seconds >= 1:  # No need to print this statistics if the total duration was too short
        print(f"There were {global_counter / duration_seconds :.2f} events per second on average.")

def window():
    """ Window display to check the code before acquiring data """
    args = parse_args()

    # Events iterator on Camera or event file
    mv_iterator = EventsIterator(input_path=args.event_file_path, delta_t=int(1e6/args.freq))	# get frequency in microseconds 

    height, width = mv_iterator.get_size()  # Camera Geometry
    with Window(title="Metavision SDK Get Started", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)

        # Event Frame Generator
        event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height,
                                                        accumulation_time_us=args.accumulation_time_us)

        def on_cd_frame_cb(ts, cd_frame):
            window.show(cd_frame)

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        global_counter = 0  # This will track how many events we processed
        global_max_t = 0  # This will track the highest timestamp we processed

        # Process events & record data in .txt file for saving
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            event_frame_gen.process_events(evs)

            print("----- New event buffer! -----")
            if evs.size == 0:
                print("The current event buffer is empty.")
            else:
                min_t = evs['t'][0]   # Get the timestamp of the first event of this callback
                max_t = evs['t'][-1]  # Get the timestamp of the last event of this callback
                global_max_t = max_t  # Events are ordered by timestamp, so the current last event has the highest timestamp

                counter = evs.size  # Local counter
                global_counter += counter  # Increase global counter
                
                print(f"There were {counter} events in this event buffer.")
                print(f"There were {global_counter} total events up to now.")
                print(f"The current event buffer included events from {min_t} to {max_t} microseconds.")
                print("----- End of the event buffer! -----")

            if window.should_close():
                break

            # Print the global statistics
            duration_seconds = global_max_t / 1.0e6
            print(f"There were {global_counter} events in total.")
            print(f"The total duration was {duration_seconds:.2f} seconds.")
            if duration_seconds >= 1:  # No need to print this statistics if the total duration was too short
                print(f"There were {global_counter / duration_seconds :.2f} events per second on average.")

if __name__ == "__main__" :
    window()
    main()

