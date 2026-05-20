#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "aikit_biz_api.h"
#include "aikit_biz_config.h"
#include "aikit_constant.h"

namespace {

constexpr const char* kAbility = "e2e44feff";

std::atomic_bool g_tts_finished(false);
std::FILE* g_pcm_file = nullptr;

void OnOutput(AIKIT_HANDLE* /*handle*/, const AIKIT_OutputData* output) {
  if (!output || !output->node || !output->node->value || !g_pcm_file) {
    return;
  }
  std::fwrite(output->node->value, sizeof(char), output->node->len, g_pcm_file);
}

void OnEvent(AIKIT_HANDLE* /*handle*/, AIKIT_EVENT event_type,
             const AIKIT_OutputEvent* /*event_value*/) {
  if (event_type == AIKIT_Event_End) {
    g_tts_finished = true;
  }
}

void OnError(AIKIT_HANDLE* /*handle*/, int32_t err, const char* desc) {
  ROS_ERROR("AIKit TTS error %d: %s", err, desc ? desc : "");
  g_tts_finished = true;
}

std::string GetEnvOrParam(const ros::NodeHandle& private_nh, const std::string& param_name,
                          const char* env_name) {
  std::string value;
  private_nh.param<std::string>(param_name, value, "");
  if (!value.empty()) {
    return value;
  }
  const char* env_value = std::getenv(env_name);
  return env_value ? std::string(env_value) : std::string();
}

class AikitSpeakNode {
 public:
  AikitSpeakNode() : private_nh_("~") {
    private_nh_.param<std::string>("speak_topic", speak_topic_, "/speak");
    private_nh_.param<std::string>("voice_name", voice_name_, "xiaoyan");
    private_nh_.param<int>("language", language_, 1);
    private_nh_.param<int>("sample_rate", sample_rate_, 16000);
    private_nh_.param<std::string>("audio_device", audio_device_, "");

    const std::string package_path = ros::package::getPath("ucar_2026_aikit_tts");
    private_nh_.param<std::string>("work_dir", work_dir_, package_path + "/third_party/aikit");
    private_nh_.param<std::string>("pcm_path", pcm_path_, "/tmp/ucar_2026_aikit_tts.pcm");

    app_id_ = GetEnvOrParam(private_nh_, "app_id", "XF_AIKIT_APPID");
    api_key_ = GetEnvOrParam(private_nh_, "api_key", "XF_AIKIT_API_KEY");
    api_secret_ = GetEnvOrParam(private_nh_, "api_secret", "XF_AIKIT_API_SECRET");
  }

  bool Init() {
    if (app_id_.empty() || api_key_.empty() || api_secret_.empty()) {
      ROS_ERROR("Missing AIKit auth. Set XF_AIKIT_APPID, XF_AIKIT_API_KEY and XF_AIKIT_API_SECRET.");
      return false;
    }

    AIKIT::AIKIT_Configurator::builder()
        .app()
        .appID(app_id_.c_str())
        .apiKey(api_key_.c_str())
        .apiSecret(api_secret_.c_str())
        .workDir(work_dir_.c_str())
        .auth()
        .authType(0)
        .log()
        .logLevel(LOG_LVL_INFO)
        .logPath(work_dir_.c_str());

    const int init_ret = AIKIT::AIKIT_Init();
    if (init_ret != 0) {
      ROS_ERROR("AIKIT_Init failed: %d", init_ret);
      return false;
    }

    AIKIT_Callbacks callbacks = {OnOutput, OnEvent, OnError};
    const int cb_ret = AIKIT::AIKIT_RegisterAbilityCallback(kAbility, callbacks);
    if (cb_ret != 0) {
      ROS_ERROR("AIKIT_RegisterAbilityCallback failed: %d", cb_ret);
      AIKIT::AIKIT_UnInit();
      return false;
    }

    sub_ = nh_.subscribe(speak_topic_, 10, &AikitSpeakNode::SpeakCallback, this);
    ROS_INFO("AIKit TTS node ready. topic=%s voice=%s work_dir=%s",
             speak_topic_.c_str(), voice_name_.c_str(), work_dir_.c_str());
    return true;
  }

  ~AikitSpeakNode() {
    AIKIT::AIKIT_UnInit();
  }

 private:
  void SpeakCallback(const std_msgs::String::ConstPtr& msg) {
    const std::string text = msg->data;
    if (text.empty()) {
      return;
    }

    std::lock_guard<std::mutex> lock(speak_mutex_);
    ROS_INFO("AIKit TTS speaking: %s", text.c_str());
    if (!SynthesizeToPcm(text)) {
      return;
    }
    PlayPcm();
  }

  bool SynthesizeToPcm(const std::string& text) {
    AIKIT::AIKIT_ParamBuilder* param_builder = nullptr;
    AIKIT::AIKIT_DataBuilder* data_builder = nullptr;
    AIKIT_HANDLE* handle = nullptr;
    AIKIT::AiText* ai_text = nullptr;
    g_tts_finished = false;

    param_builder = AIKIT::AIKIT_ParamBuilder::create();
    param_builder->clear();
    param_builder->param("vcn", voice_name_.c_str(), voice_name_.size());
    param_builder->param("vcnModel", voice_name_.c_str(), voice_name_.size());
    param_builder->param("language", language_);
    param_builder->param("textEncoding", "UTF-8", std::strlen("UTF-8"));

    int ret = AIKIT::AIKIT_Start(kAbility, AIKIT::AIKIT_Builder::build(param_builder), nullptr, &handle);
    if (ret != 0) {
      ROS_ERROR("AIKIT_Start failed: %d", ret);
      delete param_builder;
      return false;
    }

    g_pcm_file = std::fopen(pcm_path_.c_str(), "wb");
    if (!g_pcm_file) {
      ROS_ERROR("Failed to open pcm output: %s", pcm_path_.c_str());
      AIKIT::AIKIT_End(handle);
      delete param_builder;
      return false;
    }

    data_builder = AIKIT::AIKIT_DataBuilder::create();
    data_builder->clear();
    ai_text = AIKIT::AiText::get("text")->data(text.c_str(), text.size())->once()->valid();
    data_builder->payload(ai_text);

    ret = AIKIT::AIKIT_Write(handle, AIKIT::AIKIT_Builder::build(data_builder));
    if (ret != 0) {
      ROS_ERROR("AIKIT_Write failed: %d", ret);
      std::fclose(g_pcm_file);
      g_pcm_file = nullptr;
      AIKIT::AIKIT_End(handle);
      delete data_builder;
      delete param_builder;
      return false;
    }

    ros::Rate rate(1000);
    while (ros::ok() && !g_tts_finished) {
      rate.sleep();
    }

    AIKIT::AIKIT_End(handle);
    std::fclose(g_pcm_file);
    g_pcm_file = nullptr;
    delete data_builder;
    delete param_builder;
    return true;
  }

  void PlayPcm() const {
    std::string command = "aplay -q -f S16_LE -r " + std::to_string(sample_rate_) + " -c 1 ";
    if (!audio_device_.empty()) {
      command += "-D " + audio_device_ + " ";
    }
    command += pcm_path_;
    const int ret = std::system(command.c_str());
    if (ret != 0) {
      ROS_ERROR("PCM playback failed with code %d. Command: %s", ret, command.c_str());
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  ros::Subscriber sub_;
  std::mutex speak_mutex_;

  std::string speak_topic_;
  std::string voice_name_;
  std::string work_dir_;
  std::string pcm_path_;
  std::string audio_device_;
  std::string app_id_;
  std::string api_key_;
  std::string api_secret_;
  int language_ = 1;
  int sample_rate_ = 16000;
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "aikit_speak_node");
  AikitSpeakNode node;
  if (!node.Init()) {
    return 1;
  }
  ros::spin();
  return 0;
}
