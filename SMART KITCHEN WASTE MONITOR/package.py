def generate_apk_link(username, repo_name, version_tag, apk_name):
    """
    Generate a direct GitHub APK release download link
    """
    link = f"https://github.com/{username}/{repo_name}/releases/download/{version_tag}/{apk_name}"
    return link


# ====== EDIT THESE VALUES ======
github_username = "narendrakumar"          # Your GitHub username
repository_name = "smart-kitchen-waste-monitor"
version = "v1.0"
apk_filename = "app-release.apk"
# =================================

download_link = generate_apk_link(
    github_username,
    repository_name,
    version,
    apk_filename
)

print("\nYour APK Download Link:")
print(download_link)