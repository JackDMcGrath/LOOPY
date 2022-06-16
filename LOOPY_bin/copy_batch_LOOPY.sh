#!/bin/bash
echo ""
echo "Copy batch_LiCSBAS.sh to the current directory."
echo "(cp -i $LOOPY_PATH/batch_LOOPY.sh .; chmod 755 batch_LOOPY.sh)"
cp -i $LOOPY_PATH/batch_LOOPY.sh .
chmod 755 batch_LOOPY.sh

echo ""
echo "Edit batch_LOOPY.sh, then run ./batch_LOOPY.sh"
echo ""
