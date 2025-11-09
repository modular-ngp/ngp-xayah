#include <args.hxx>
#include <vector>
#include <string>
#include <iostream>

struct ArgParserPack {
    args::ArgumentParser parser;
    args::HelpFlag help_flag;
    args::ValueFlag<std::string> mode_flag;
    args::ValueFlag<std::string> network_config_flag;
    args::Flag no_gui_flag;
    args::Flag vr_flag;
    args::Flag no_train_flag;
    args::ValueFlag<std::string> scene_flag;
    args::ValueFlag<std::string> snapshot_flag;
    args::ValueFlag<uint32_t> width_flag;
    args::ValueFlag<uint32_t> height_flag;
    args::Flag version_flag;
    args::PositionalList<std::string> files;

    ArgParserPack()
        : parser("Instant Neural Graphics Primitives: Version 1.0.0\n"),
          help_flag(parser, "HELP", "Display this help menu.", {'h', "help"}),
          mode_flag(parser, "MODE", "Deprecated. Do not use.", {'m', "mode"}),
          network_config_flag(parser, "CONFIG", "Path to the network config.", {'n', 'c', "network", "config"}),
          no_gui_flag(parser, "NO_GUI", "Disables the GUI.", {"no-gui"}),
          vr_flag(parser, "VR", "Enables VR.", {"vr"}),
          no_train_flag(parser, "NO_TRAIN", "Disables training on startup.", {"no-train"}),
          scene_flag(parser, "SCENE", "The scene to load.", {'s', "scene"}),
          snapshot_flag(parser, "SNAPSHOT", "Optional snapshot to load.", {"snapshot", "load_snapshot"}),
          width_flag(parser, "WIDTH", "Resolution width of the GUI.", {"width"}),
          height_flag(parser, "HEIGHT", "Resolution height of the GUI.", {"height"}),
          version_flag(parser, "VERSION", "Display the version.", {'v', "version"}),
          files(parser, "files", "Files to be loaded.") {}
};

int main(int argc, char* argv[]) {
    std::vector<std::string> arguments(argv, argv + argc);

    ArgParserPack ap;
    auto &parser = ap.parser;

    if (arguments.empty()) {
        std::cerr << "Number of arguments must be bigger than 0." << std::endl;
        return -3;
    }

    try {
        parser.Prog(arguments.front());
        parser.ParseArgs(std::begin(arguments) + 1, std::end(arguments));
    } catch (const args::Help&) {
        std::cout << parser;
        return 0;
    } catch (const args::ParseError& e) {
        std::cerr << e.what() << "\n" << parser;
        return -1;
    } catch (const args::ValidationError& e) {
        std::cerr << e.what() << "\n" << parser;
        return -2;
    }
    return 0;
}
