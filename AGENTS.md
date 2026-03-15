# Agent Development Guidelines for parakeet-server

## Build, Lint, and Test Commands

### Building the Project
```bash
# Build the release version
cargo build --release

# Build with debug info
cargo build

# Build and run in debug mode
cargo run
```

### Running Tests
```bash
# Run all tests (will run in release mode)
cargo test

# Run a specific test function by name
cargo test <test_function_name>

# Run tests with verbose output
cargo test --verbose

# Run tests in release mode
cargo test --release
```

### Linting and Formatting
```bash
# Format the code (requires rustfmt)
cargo fmt

# Check formatting without modifying files
cargo fmt --check

# Lint the code (requires clippy)
cargo clippy

# Run clippy with warnings as errors
cargo clippy -- -D warnings
```

## Code Style Guidelines

### Imports
- Group imports in order: standard library, external crates, local modules
- Use fully qualified paths for external dependencies
- Organize imports alphabetically within each group
- Avoid unused imports
- Keep import statements clean and readable, no wildcards `*` unless necessary

### Formatting
- Use `cargo fmt` for code formatting with default Rust style
- Follow the standard Rust naming conventions and code structure
- Limit lines to 100 characters
- Use proper spacing and indentation (4 spaces)
- Use camelCase for function names, snake_case for variables
- Use PascalCase for types

### Types
- Use explicit type annotations where helpful for clarity
- Prefer `Option<T>` over `T` when a value might be absent
- Prefer `Result<T, E>` for operations that can fail
- Use `Vec<T>` for growable arrays, `&[T]` for slices when appropriate
- Avoid using `String` type in favor of `&str` when possible 
- Use meaningful names for types to clearly communicate their purpose

### Naming Conventions
- Use snake_case for functions, variables, and modules
- Use PascalCase for types and enums
- Use UPPER_CASE for constants
- Use descriptive names for functions and variables
- Avoid abbreviations unless they are widely known
- Prefix private functions with underscore if appropriate

### Error Handling
- Handle errors explicitly using Result and Option types
- Prefer early returns (`?` operator) over nested `match` statements where possible
- Provide meaningful error messages
- Log errors appropriately using the provided `log` crate
- Use appropriate error types from Rust's standard library or external crates where applicable

### Logging
Use the `log` crate with appropriate log levels:
- `log::info!()` for informational messages 
- `log::warn!()` for warnings
- `log::error!()` for errors
- `log::debug!()` for debugging information (should be hidden in release builds)

### Documentation
- Document all public functions with doc comments using /// syntax
- Use `#[doc(hidden)]` attribute to hide private implementation details from documentation
- Add comments to explain complex logic or algorithmic decisions
- Include examples in documentation where appropriate

## Pull Request Guidelines

### PR Title Format

Start with an action verb in present tense:
- "Add" for new features
- "Update" or "Refactor" for modifications
- "Fix" or "Remove" for bug fixes/cleanup
- Example: "Add Parakeet Bench command for transcription performance benchmarking"

### PR Description Format

Use the following structure when creating a PR description:

**Summary:** An explanation of what the PR does and why it's needed.

**Key Changes:** A bulleted list of the specific modifications made (new files, modified files, key logic changes).

**Testing Notes:** Technical details on how the changes were verified or tested. Include any manual steps required or inferred from the diff and commits.

### Creating PRs with GitHub CLI

Use `gh pr create` to create detailed PRs:

```bash
# Create new branch first if needed
git checkout -b feature-name

# Make changes, commit them
git add . && git commit -m "Short commit message"

# Create PR from terminal
gh pr create --base master \
  --title "Title here" \
  --body $'## Summary\n\nDescription...\n\n## Key Changes\n\n- Change 1\n- Change 2\n\n## Testing Notes\n\nNotes...'
```

Then open in browser: `gh pr view --web`

---

## Additional Configuration Files

The project uses:
- `.gitignore` - standard Git ignore patterns
- `Dockerfile` - for containerized builds
- `run.sh` - convenience script to start the server
- `test_transcribe.sh` - testing script for audio transcription
- `Rocket.toml` - Rocket web framework configuration (minimal)