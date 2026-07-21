REMOTE_HOST="mehrot@console2.ghpc.au.dk"
REMOTE_BASE="/usr/home/qgg/mehrot/fish-data"
LOCAL_BASE="/home/devd/fish/data"

cd "$LOCAL_BASE" && find . -type f -print0 | \
parallel --ungroup -0 -j 8 --pipe --block 100M \
  rsync -avh --info=progress2 --ignore-existing \
  --from0 --files-from=- \
  "$LOCAL_BASE/" \
  "$REMOTE_HOST:$REMOTE_BASE/"
  