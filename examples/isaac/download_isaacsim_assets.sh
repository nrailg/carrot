#!/usr/bin/env bash
set -Eeuo pipefail

# DFS 为当前用户的个人持久化目录，由调用方明确设置。
: "${DFS:?请先 export DFS 为当前用户的个人 DFS 绝对路径}"
[[ "$DFS" = /* ]] || { echo 'DFS 必须是绝对路径' >&2; exit 1; }
[[ "$#" -eq 0 ]] || { echo '不接受位置参数，请通过 DFS 设置存储根目录' >&2; exit 1; }
ASSET_DIR="${DFS%/}/isaacsim_assets"
DOWNLOAD_DIR="$ASSET_DIR/downloads/6.0.0"
ASSET_ROOT="$ASSET_DIR/Assets/Isaac/6.0"
BASE_URL='https://downloads.isaacsim.nvidia.com'
# 官方 6.0 完整包是一个 ZIP 的五个分片，必须按此顺序合并。
PARTS=(
  'isaac-sim-assets-complete-6.0.0.001.zip'
  'isaac-sim-assets-complete-6.0.0.002.zip'
  'isaac-sim-assets-complete-6.0.0.003.zip'
  'isaac-sim-assets-complete-6.0.0.004.zip'
  'isaac-sim-assets-complete-6.0.0.005.zip'
)
MD5=(
  '401e58c4e08c906fab5fc6fa6825c1cb'
  '201941c1f0cdc91346cc40a941d8afaf'
  '8cf4da965aed1a1eca5a9362f689bda8'
  '14c023814d805c927e9c8cf766213ee1'
  'ab770e11d0365c6b4a3591caf5daf5bb'
)

for tool in curl md5sum unzip; do
  command -v "$tool" >/dev/null || { printf '缺少依赖：%s\n' "$tool" >&2; exit 1; }
done

# 按 wx-download 设置 Gemini 下载代理。
cat > /tmp/set_proxy.sh <<'PROXY'
export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"
PROXY
source /tmp/set_proxy.sh

mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"
for i in "${!PARTS[@]}"; do
  part="${PARTS[$i]}"
  if [[ -f "$part" ]]; then
    printf '%s  %s\n' "${MD5[$i]}" "$part" | md5sum --check - || {
      printf '缓存校验失败，请移走 %s 后重跑。\n' "$DOWNLOAD_DIR/$part" >&2
      exit 1
    }
    continue
  fi
  printf '下载分片 %s（支持断点续传）\n' "$part"
  curl --fail --location --show-error --retry 5 --connect-timeout 30 \
    --continue-at - --output "$part.part" "$BASE_URL/$part"
  printf '%s  %s\n' "${MD5[$i]}" "$part.part" | md5sum --check - || {
    printf '下载校验失败，请移走 %s 后重跑。\n' "$DOWNLOAD_DIR/$part.part" >&2
    exit 1
  }
  mv -- "$part.part" "$part"
done

ARCHIVE='isaac-sim-assets-complete-6.0.0.zip'
printf '按顺序合并五个分片：%s\n' "$ARCHIVE"
cat "${PARTS[@]}" > "$ARCHIVE.part"
mv -- "$ARCHIVE.part" "$ARCHIVE"
# unzip 会校验每个解压文件的 CRC；-o 允许补齐此前已有的部分资源。
unzip -o "$ARCHIVE" -d "$ASSET_DIR"

[[ -d "$ASSET_ROOT/Isaac" && -d "$ASSET_ROOT/NVIDIA" ]]
[[ -f "$ASSET_ROOT/Isaac/Environments/Grid/default_environment.usd" ]]
[[ -f "$ASSET_ROOT/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd" ]]
[[ -f "$ASSET_ROOT/Isaac/Props/Mounts/SeattleLabTable/table.usd" ]]
# source 此文件后，示例的 asset_root.py 会覆盖 Isaac Lab 的资源常量。
printf 'export ISAACSIM_ASSET_ROOT=%q\nexport OMNI_KIT_ACCEPT_EULA=YES\n' \
  "$ASSET_ROOT" > "$ASSET_DIR/asset_env.sh"
printf '\n完整资源已解压，无需模型转换。运行前执行：\nsource %q\n' "$ASSET_DIR/asset_env.sh"
printf '分片和合并 ZIP 保留在 %s，可在确认成功后自行清理。\n' "$DOWNLOAD_DIR"
