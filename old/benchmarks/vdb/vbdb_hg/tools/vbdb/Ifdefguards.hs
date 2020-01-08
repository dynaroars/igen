
module Ifdefguards
  ( findIfdefPatterns )
  where

import Note ( Pc, parsePc )

import qualified Data.ByteString.Char8 as BC
import           System.IO ( hGetContents )
import           System.Process
                 ( CreateProcess(..), StdStream(..)
                 , createProcess, shell
                 , waitForProcess
                 )

findIfdefPatterns :: FilePath -> IO [Pc]
findIfdefPatterns benchsDir = grepIfdefs benchsDir >>= return . map parseLine . lines

grepIfdefs :: FilePath -> IO String
grepIfdefs benchsDir = do
  (_,Just out,_,ph) <- createProcess cmd{std_out=CreatePipe}
  waitForProcess ph
  hGetContents out
  where cmd = shell $ "egrep \"#(el)?if\" -R "
                        ++ benchsDir ++ "/*.c"

parseLine :: String -> Pc
parseLine = parsePc . BC.pack . takeGuard . stripLoc
  where stripLoc = dropWhile (/= '#')
        takeGuard x = case break (== ' ') x of
          ("#ifndef",' ':guard)  -> '!' : guard
          (__________,' ':guard) -> guard
          _other                 -> error "Ifdefguards: parseLine"
