{-# LANGUAGE BangPatterns #-}

{- Limitations
We assume that there will be a "SUM:" line in the output of cloc, otherwise
the folder is reported as empty. However, if the folder contains code from one language only, there will be no "SUM:" line. Thus, any "EMPTY" folder must be
manually examined.

TODO:
Alternatively we could take the last line in the output ? To be sure that some
output was produced we need to find a line of dashes.
-}

module LocSize ( showStats ) where

import           Control.Monad ( filterM, forM )
import           Data.Decimal ( Decimal, decimalMantissa, roundTo )
import           Data.Functor ( (<$>) )
import           Data.List (  isPrefixOf )
import           System.Directory ( doesDirectoryExist, getDirectoryContents )
import           System.FilePath
import           System.Process ( readProcess )

type KLOC = Decimal

showKLOC :: KLOC -> String
showKLOC = show . roundTo 1

cloc :: FilePath -> FilePath -> IO String
cloc src dir = readProcess "/usr/bin/cloc" [ "--quiet", src </> dir ] ""

readKLOC :: String -> KLOC
readKLOC str | null sumLines = 0
             | otherwise     = fromInteger locCount / 1000
  where sumLines = filter ("SUM:" `isPrefixOf`) $ lines str
        sumLine = head sumLines
        locCount :: Integer
        locCount = read $ last $ words sumLine

findKLOC :: FilePath -> FilePath -> IO KLOC
findKLOC src dir = readKLOC <$> cloc src dir

findSubsystems :: FilePath -> IO [FilePath]
findSubsystems src = do
    contents <- filter (`notElem` nopeList) <$> getDirectoryContents src
    filterM isSubsystem contents
    where nopeList = [".","..",".git"]
          isSubsystem x = doesDirectoryExist (src </> x)


showStats :: FilePath -> IO String
showStats src = do
    !totalLoc <- findKLOC src "."
    subsystems <- findSubsystems src
    sysLines <- forM subsystems $ \sys -> do
        sysLoc <- findKLOC src sys
        return $  "\n# " ++ sys ++ "\n" ++
            if sysLoc > 0 then
                   "KLOC = " ++ showKLOC sysLoc ++ "\n"
                ++ "%size= " ++ show (percent sysLoc totalLoc) ++ "\n"
                ++ "[]'s = " ++ show (squares sysLoc) ++ "\n"
                else "EMPTY\n"
    return $ "Total = " ++ showKLOC totalLoc ++ " KLOC\n"
             ++ unlines sysLines
    where percent :: KLOC -> KLOC -> Decimal
          percent x tot = roundTo 1 $ 100*x / tot
          squares :: KLOC -> Integer
          squares x = decimalMantissa $ roundTo 0 (x / 0.5)
--           squares x = decimalMantissa $ roundTo 0 (x / 25)
