#include <args.hxx>

int main(int argc, char *argv[]) {
    std::vector<std::string> arguments;
    for (int i = 0; i < argc; ++i) {
        arguments.emplace_back(argv[i]);
    }

    args::ArgumentParser parser{"Instant Neural Graphics Primitives: Version 1.0.0\n"};

    args::HelpFlag help_flag{
        parser,
        "HELP",
        "Display this help menu.",
        {'h', "help"},
    };

    args::ValueFlag<std::string> mode_flag{
        parser,
        "MODE",
        "Deprecated. Do not use.",
        {'m', "mode"},
    };

    args::ValueFlag<std::string> network_config_flag{
        parser,
        "CONFIG",
        "Path to the network config. Uses the scene's default if unspecified.",
        {'n', 'c', "network", "config"},
    };

    args::Flag no_gui_flag{
        parser,
        "NO_GUI",
        "Disables the GUI and instead reports training progress on the command line.",
        {"no-gui"},
    };

    args::Flag vr_flag{
        parser,
        "VR",
        "Enables VR",
        {"vr"}
    };

    args::Flag no_train_flag{
        parser,
        "NO_TRAIN",
        "Disables training on startup.",
        {"no-train"},
    };

    args::ValueFlag<std::string> scene_flag{
        parser,
        "SCENE",
        "The scene to load. Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume.",
        {'s', "scene"},
    };

    args::ValueFlag<std::string> snapshot_flag{
        parser,
        "SNAPSHOT",
        "Optional snapshot to load upon startup.",
        {"snapshot", "load_snapshot"},
    };

    args::ValueFlag<uint32_t> width_flag{
        parser,
        "WIDTH",
        "Resolution width of the GUI.",
        {"width"},
    };

    args::ValueFlag<uint32_t> height_flag{
        parser,
        "HEIGHT",
        "Resolution height of the GUI.",
        {"height"},
    };

    args::Flag version_flag{
        parser,
        "VERSION",
        "Display the version of instant neural graphics primitives.",
        {'v', "version"},
    };

    args::PositionalList<std::string> files{
        parser,
        "files",
        "Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.",
    };

    try {
        if (arguments.empty()) {
            std::cerr << "Number of arguments must be bigger than 0." << std::endl;
            return -3;
        }

        parser.Prog(arguments.front());
        parser.ParseArgs(std::begin(arguments) + 1, std::end(arguments));
    } catch (const args::Help &) {
        std::cout << parser;
        return 0;
    } catch (const args::ParseError &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return -1;
    } catch (const args::ValidationError &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return -2;
    }
    return 0;
}
