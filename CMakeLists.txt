cmake_minimum_required(VERSION 3.16)
project(nnQt LANGUAGES CXX)

# Konfigurasi dasar Qt
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Temukan Qt6
find_package(Qt6 REQUIRED COMPONENTS Widgets Core)

# Konfigurasi pencarian Rust library
set(RUST_LIB_NAME "rustcore") # Sesuaikan dengan nama di Cargo.toml
set(RUST_BUILD_MODE "release") # Ganti ke "debug" jika perlu

# Cari library Rust dengan fallback ke debug/release
find_library(RUST_LIB
    NAMES 
        ${RUST_LIB_NAME}
        lib${RUST_LIB_NAME}.so    # Linux
        lib${RUST_LIB_NAME}.dylib # macOS
        ${RUST_LIB_NAME}.dll      # Windows
    PATHS
        "${CMAKE_SOURCE_DIR}/../rust/target/${RUST_BUILD_MODE}"
        "${CMAKE_SOURCE_DIR}/../target/${RUST_BUILD_MODE}"
        "${CMAKE_SOURCE_DIR}/../../rust/target/${RUST_BUILD_MODE}"
    NO_DEFAULT_PATH
)

if(NOT RUST_LIB)
    # Fallback: Coba cari di seluruh sistem
    find_library(RUST_LIB
        NAMES 
            ${RUST_LIB_NAME}
            lib${RUST_LIB_NAME}
    )
    
    if(NOT RUST_LIB)
        message(FATAL_ERROR "Rust library (${RUST_LIB_NAME}) tidak ditemukan!\n"
            "Pastikan:\n"
            "1. Rust project sudah dibuild dengan 'cargo build --release'\n"
            "2. Nama library di Cargo.toml sesuai\n"
            "3. Lokasi library benar di: ${CMAKE_SOURCE_DIR}/../rust/target/${RUST_BUILD_MODE}/")
    endif()
endif()

message(STATUS "Found Rust library: ${RUST_LIB}")

# File sumber
set(SOURCES
    main.cpp
    mainwindow.cpp
)

set(HEADERS
    mainwindow.h
)

set(UIS
    mainwindow.ui
)

# Konfigurasi executable
add_executable(nnQt
    ${SOURCES}
    ${HEADERS}
    ${UIS}
)

# Link libraries
target_link_libraries(nnQt PRIVATE
    Qt6::Widgets
    Qt6::Core
    ${RUST_LIB}
)

# Include directories
target_include_directories(nnQt PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/../rust/src  # Sesuaikan dengan path header Rust jika ada
)

# Copy Rust library ke build directory (opsional)
add_custom_command(TARGET nnQt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${RUST_LIB}
        $<TARGET_FILE_DIR:nnQt>
)