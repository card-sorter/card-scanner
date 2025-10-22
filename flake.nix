{
  description = "Python development environment with uv";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python313
            uv  # Add uv to the build inputs
            stdenv.cc.cc
            zlib
            fuse3
            icu
            nss
            openssl
            curl
            expat
            xorg.libX11
            vulkan-headers
            vulkan-loader
            vulkan-tools
          ];
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              pkgs.stdenv.cc.cc
              pkgs.zlib
              pkgs.fuse3
              pkgs.icu
              pkgs.nss
              pkgs.openssl
              pkgs.curl
              pkgs.expat
              pkgs.xorg.libX11
              pkgs.vulkan-headers
              pkgs.vulkan-loader
              pkgs.vulkan-tools
            ]}:$LD_LIBRARY_PATH
            # Create a virtual environment if it doesn't exist
            if [ ! -d ".venv" ]; then
              uv venv .venv
            fi
            # Activate the virtual environment
            source .venv/bin/activate
            # Alias pip to uv for faster package installation
            alias pip="uv pip"
          '';
        };
      }
    );
}