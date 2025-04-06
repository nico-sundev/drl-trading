import sys
import os
from string import Template


# ==== CONFIGURATION ====

FEATURE_PACKAGE = "ai_trading/preprocess/feature/collection"
TEST_PACKAGE = "tests/preprocess/feature/collection"
TEMPLATE_DIR = "templates"  # Folder where your templates live

# Template filenames (you'll provide these)
FEATURE_TEMPLATE_FILE = "feature_template.txt"
TEST_TEMPLATE_FILE = "feature_test_template.txt"
CONFIG_TEMPLATE_FILE = "config_template.txt"

# ========================


def snake_to_pascal(snake: str) -> str:
    return ''.join(word.capitalize() for word in snake.split('_'))


def render_template(path: str, substitutions: dict) -> str:
    with open(path, 'r') as f:
        template = Template(f.read())
    return template.substitute(substitutions)


def create_file_from_template(output_path: str, template_path: str, substitutions: dict):
    content = render_template(template_path, substitutions)
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Created: {output_path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python add_feature.py <feature_name>")
        sys.exit(1)

    feature_name = sys.argv[1].lower()
    class_prefix = snake_to_pascal(feature_name)

    substitutions = {
        "feature_name": feature_name,
        "FeatureName": class_prefix
    }

    # 1. Create Feature class
    feature_path = os.path.join(FEATURE_PACKAGE, f"{feature_name}_feature.py")
    feature_template = os.path.join(TEMPLATE_DIR, FEATURE_TEMPLATE_FILE)
    create_file_from_template(feature_path, feature_template, substitutions)

    # 2. Create Unit Test
    test_path = os.path.join(TEST_PACKAGE, f"test_{feature_name}_feature.py")
    test_template = os.path.join(TEMPLATE_DIR, TEST_TEMPLATE_FILE)
    create_file_from_template(test_path, test_template, substitutions)

    # 3. Print Config to terminal
    config_template = os.path.join(TEMPLATE_DIR, CONFIG_TEMPLATE_FILE)
    config_code = render_template(config_template, substitutions)
    print("\n=== Suggested Config Class ===")
    print(config_code)


if __name__ == "__main__":
    main()
