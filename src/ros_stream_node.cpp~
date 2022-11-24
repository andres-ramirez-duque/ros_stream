// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.

#include "whisper.h"

// third-party utilities
// use your favorite implementations
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <SDL.h>
#include <SDL_audio.h>

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>

#include <ros/ros.h>
#include <std_srvs/Trigger.h>
//  500 -> 00:05.000
// 6000 -> 01:00.000

SDL_AudioDeviceID g_dev_id_in = 0;
const int n_samples_30s = 30*WHISPER_SAMPLE_RATE;
const int n_samples_keep = 0.2*WHISPER_SAMPLE_RATE;
int n_samples;
int n_samples_len;

int n_iter = 0;

whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
std::vector<whisper_token> prompt_tokens;

struct whisper_context * ctx;
    
std::vector<float> pcmf32(n_samples_30s, 0.0f);
std::vector<float> pcmf32_old;
    
std::string to_timestamp(int64_t t) {
    int64_t sec = t/100;
    int64_t msec = t - sec*100;
    int64_t min = sec/60;
    sec = sec - min*60;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d.%03d", (int) min, (int) sec, (int) msec);

    return std::string(buf);
}

// command-line parameters
struct whisper_params {
    int32_t seed       = -1; // RNG seed, not used currently
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    bool speed_up             = false;
    bool verbose              = false;
    bool translate            = false;
    bool no_context           = true;
    bool print_special_tokens = false;
    bool no_timestamps        = true;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out = "";
};

whisper_params params;

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "--step") {
            params.step_ms = std::stoi(argv[++i]);
        } else if (arg == "--length") {
            params.length_ms = std::stoi(argv[++i]);
        } else if (arg == "-c" || arg == "--capture") {
            params.capture_id = std::stoi(argv[++i]);
        } else if (arg == "-mt" || arg == "--max_tokens") {
            params.max_tokens = std::stoi(argv[++i]);
        } else if (arg == "-ac" || arg == "--audio_ctx") {
            params.audio_ctx = std::stoi(argv[++i]);
        } else if (arg == "-su" || arg == "--speed-up") {
            params.speed_up = true;
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "--translate") {
            params.translate = true;
        } else if (arg == "-kc" || arg == "--keep-context") {
            params.no_context = false;
        } else if (arg == "-l" || arg == "--language") {
            params.language = argv[++i];
            if (whisper_lang_id(params.language.c_str()) == -1) {
                fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
                whisper_print_usage(argc, argv, params);
                exit(0);
            }
        } else if (arg == "-ps" || arg == "--print_special") {
            params.print_special_tokens = true;
        } else if (arg == "-nt" || arg == "--no_timestamps") {
            params.no_timestamps = true;
        } else if (arg == "-m" || arg == "--model") {
            params.model = argv[++i];
        } else if (arg == "-f" || arg == "--file") {
            params.fname_out = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int argc, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help           show this help message and exit\n");
    fprintf(stderr, "  -s SEED,  --seed SEED      RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N,     --threads N      number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "            --step N         audio step size in milliseconds (default: %d)\n", params.step_ms);
    fprintf(stderr, "            --length N       audio length in milliseconds (default: %d)\n", params.length_ms);
    fprintf(stderr, "  -c ID,    --capture ID     capture device ID (default: -1)\n");
    fprintf(stderr, "  -mt N,    --max_tokens N   maximum number of tokens per audio chunk (default: %d)\n", params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio_ctx N    audio context size (default: %d, 0 - all)\n", params.audio_ctx);
    fprintf(stderr, "  -su,      --speed-up       speed up audio by factor of 2 (faster processing, reduced accuracy, default: %s)\n", params.speed_up ? "true" : "false");
    fprintf(stderr, "  -v,       --verbose        verbose output\n");
    fprintf(stderr, "            --translate      translate from source language to english\n");
    fprintf(stderr, "  -kc,      --keep-context   keep text context from earlier audio (default: false)\n");
    fprintf(stderr, "  -ps,      --print_special  print special tokens\n");
    fprintf(stderr, "  -nt,      --no_timestamps  do not print timestamps\n");
    fprintf(stderr, "  -l LANG,  --language LANG  spoken language (default: %s)\n", params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME    model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME     text output file name (default: no output to file)\n");
    fprintf(stderr, "\n");
}

//
// SDL Audio Init
//

bool audio_sdl_init(const int capture_id) {
    if (g_dev_id_in) {
        fprintf(stderr, "%s: already initialized\n", __func__);
        return false;
    }

    if (g_dev_id_in == 0) {
        SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);

        if (SDL_Init(SDL_INIT_AUDIO) < 0) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s\n", SDL_GetError());
            return (1);
        }

        SDL_SetHintWithPriority(SDL_HINT_AUDIO_RESAMPLING_MODE, "medium", SDL_HINT_OVERRIDE);

        {
            int nDevices = SDL_GetNumAudioDevices(SDL_TRUE);
            fprintf(stderr, "%s: found %d capture devices:\n", __func__, nDevices);
            for (int i = 0; i < nDevices; i++) {
                fprintf(stderr, "%s:    - Capture device #%d: '%s'\n", __func__, i, SDL_GetAudioDeviceName(i, SDL_TRUE));
            }
        }
    }

    if (g_dev_id_in == 0) {
        SDL_AudioSpec capture_spec_requested;
        SDL_AudioSpec capture_spec_obtained;

        SDL_zero(capture_spec_requested);
        SDL_zero(capture_spec_obtained);

        capture_spec_requested.freq     = WHISPER_SAMPLE_RATE;
        capture_spec_requested.format   = AUDIO_F32;
        capture_spec_requested.channels = 1;
        capture_spec_requested.samples  = 1024;

        if (capture_id >= 0) {
            fprintf(stderr, "%s: attempt to open capture device %d : '%s' ...\n", __func__, capture_id, SDL_GetAudioDeviceName(capture_id, SDL_TRUE));
            g_dev_id_in = SDL_OpenAudioDevice(SDL_GetAudioDeviceName(capture_id, SDL_TRUE), SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);
        } else {
            fprintf(stderr, "%s: attempt to open default capture device ...\n", __func__);
            g_dev_id_in = SDL_OpenAudioDevice(nullptr, SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);
        }
        if (!g_dev_id_in) {
            fprintf(stderr, "%s: couldn't open an audio device for capture: %s!\n", __func__, SDL_GetError());
            g_dev_id_in = 0;
        } else {
            fprintf(stderr, "%s: obtained spec for input device (SDL Id = %d):\n", __func__, g_dev_id_in);
            fprintf(stderr, "%s:     - sample rate:       %d\n", __func__, capture_spec_obtained.freq);
            fprintf(stderr, "%s:     - format:            %d (required: %d)\n", __func__, capture_spec_obtained.format, capture_spec_requested.format);
            fprintf(stderr, "%s:     - channels:          %d (required: %d)\n", __func__, capture_spec_obtained.channels, capture_spec_requested.channels);
            fprintf(stderr, "%s:     - samples per frame: %d\n", __func__, capture_spec_obtained.samples);
        }
    }


    return true;
}
//
// SDL Audio capture
//
bool audio_sdl_capture() {

  // process new audio
  if (n_iter > 0 && SDL_GetQueuedAudioSize(g_dev_id_in) > 2*n_samples*sizeof(float)) {
    fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
    SDL_ClearQueuedAudio(g_dev_id_in);
  }

        while (SDL_GetQueuedAudioSize(g_dev_id_in) < n_samples*sizeof(float) && ros::ok()) {
            SDL_Delay(1);
        }

        const int n_samples_new = SDL_GetQueuedAudioSize(g_dev_id_in)/sizeof(float);

        // take one second from previous iteration
        //const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_30s/30 - n_samples_new));

        // take up to params.length_ms audio from previous iteration
        const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

        //printf("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());

        pcmf32.resize(n_samples_new + n_samples_take);

        for (int i = 0; i < n_samples_take; i++) {
            pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
        }

        SDL_DequeueAudio(g_dev_id_in, pcmf32.data() + n_samples_take, n_samples_new*sizeof(float));

        pcmf32_old = pcmf32;
   return true;
}

//
// Whisper ROS Service Callback
//
bool whisper_listener_callback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res)
{
  printf("[Start speaking] \n");
  SDL_PauseAudioDevice(g_dev_id_in, 0);
  
  if (!audio_sdl_capture()) {
    fprintf(stderr, "%s: audio_sdl_capture() failed!\n", __func__);
    return 1;
  }
  // run the inference
  wparams.prompt_tokens        = params.no_context ? nullptr : prompt_tokens.data();
  wparams.prompt_n_tokens      = params.no_context ? 0       : prompt_tokens.size();
  
  if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
    fprintf(stderr, "Inference failed to process audio\n");
    return 6;
  }

  const int n_segments = whisper_full_n_segments(ctx);
  for (int i = 0; i < n_segments; ++i) {
    const char * text = whisper_full_get_segment_text(ctx, i);
    if (params.no_timestamps) {
      //printf("%s \n", text);
      res.message += text;
    }
  }
  ROS_INFO("[%s\n]",res.message.c_str());
//*************************************
  ++n_iter;
  // keep part of the audio for next iteration to try to mitigate word boundary issues
  pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());
  // Add tokens of the last full length segment as the prompt
  if (!params.no_context) {
    prompt_tokens.clear();
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
      const int token_count = whisper_full_n_tokens(ctx, i);
      for (int j = 0; j < token_count; ++j) {
        prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
      }
    }
  }
  SDL_PauseAudioDevice(g_dev_id_in, 1);
  res.success = true;
  return true;
}
///////////////////////////

int main(int argc, char ** argv) {
    ros::init(argc, argv, "ros_whisper_stream_node");
    ros::NodeHandle n;
    ros::Rate r(10); // 10 hz
    ros::ServiceServer service = n.advertiseService("whisper_listener", whisper_listener_callback);
    ROS_INFO("Ready to start whisper listener service");

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    // init audio

    if (!audio_sdl_init(params.capture_id)) {
        fprintf(stderr, "%s: audio_sdl_init() failed!\n", __func__);
        return 1;
    }

    // whisper init

    ctx = whisper_init(params.model.c_str());

    n_samples = (params.step_ms/1000.0)*WHISPER_SAMPLE_RATE;
    n_samples_len = (params.length_ms/1000.0)*WHISPER_SAMPLE_RATE;
    
    const int n_new_line = params.length_ms / params.step_ms - 1;
    // the below info was previously on inference loop
    
    wparams.print_progress       = false;
    wparams.print_special_tokens = params.print_special_tokens;
    wparams.print_realtime       = false;
    wparams.print_timestamps     = !params.no_timestamps;
    wparams.translate            = params.translate;
    wparams.no_context           = true;
    wparams.single_segment       = true;
    wparams.max_tokens           = params.max_tokens;
    wparams.language             = params.language.c_str();
    wparams.n_threads            = params.n_threads;

    wparams.audio_ctx            = params.audio_ctx;
    wparams.speed_up             = params.speed_up;

    // print some info about the processing
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing %d samples (step = %.1f sec / len = %.1f sec), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                n_samples,
                float(n_samples)/WHISPER_SAMPLE_RATE,
                float(n_samples_len)/WHISPER_SAMPLE_RATE,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        fprintf(stderr, "%s: n_new_line = %d\n", __func__, n_new_line);
        fprintf(stderr, "\n");
    }

    bool is_running = true;

/******************************* [Start speaking] **********************************/

    // main audio loop
    while (is_running && ros::ok()) {
        // process SDL events:
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    {
                        is_running = false;
                    } break;
                default:
                    break;
            }
        }

        if (!is_running) {
            break;
        }
        // capture audio
        ros::spinOnce();
        r.sleep();
    }
    /******************************* [end loop speaking] **********************************/
    whisper_print_timings(ctx);
    whisper_free(ctx);
    return 0;
}
