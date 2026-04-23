#!/bin/bash
set -eu

echo '# Your Additional Custom Nodes Dependencies' > additional-pak.txt

array=(
    # 3d trellies.2
    https://github.com/visualbruno/ComfyUI-Trellis2/raw/refs/heads/main/requirements.txt
  
)

for line in "${array[@]}";
    do curl -w "\n" -sSL "${line}" >> additional-pak.txt
done

# 标准化处理（与generate-pak5.sh保持一致）
sed -i '/^#/d' additional-pak.txt
sed -i 's/[[:space:]]*$//' additional-pak.txt
sed -i 's/>=.*$//' additional-pak.txt
sed -i 's/_/-/g' additional-pak.txt

echo "Additional dependencies generated to additional-pak.txt"