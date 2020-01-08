
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

module Note where

import Params

import           Control.Monad ( (>=>), forM_ )
import           Data.ByteString ( ByteString )
import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as BC
import           Data.List ( intercalate, isSuffixOf, nub, sort )
import qualified Data.List.Split as L
import           Data.Yaml.YamlLight as Y
import           System.Directory ( getDirectoryContents )
import           System.FilePath ( (</>), splitPath )

-- * Actions on notes



isNoteFile :: FilePath -> Bool
isNoteFile fp = ".bug" `isSuffixOf` fp

isBenchFile :: FilePath -> Bool
isBenchFile fp = ".c" `isSuffixOf` fp

onFilesM :: FilePath -> (FilePath -> Bool) -> ([FilePath] -> IO a) -> IO a
onFilesM dir sel f = f =<< return . filter sel
                       =<< getDirectoryContents dir

onNotesM :: FilePath -> ([Note] -> IO a) -> IO a
onNotesM notesPath f = onFilesM notesPath isNoteFile $ \fps -> f =<< mapM (readNote notesPath) fps

onNotesPrintM :: PP t => FilePath -> ([Note] -> IO t) -> IO ()
onNotesPrintM notesPath f = onNotesM notesPath (f >=> putStrLn . pp)

onNotesPrint :: PP t => FilePath -> ([Note] -> t) -> IO ()
onNotesPrint notesPath f = onNotesM notesPath (putStrLn . pp . f)

onNotes :: FilePath -> ([Note] -> a) -> IO a
onNotes notesPath f = onNotesM notesPath (return . f)

onBenchsM :: ([FilePath] -> IO a) -> IO a
onBenchsM = onFilesM paramBENCHSDIR isBenchFile

-- * Representation

data Note =
  Note {
    noteHash   :: !Hash,
    noteType   :: !Type,
    notePc     :: !Pc,
    noteFixed  :: ![FixedIn],
    noteLoc    :: !FilePath
  }
  deriving Show

type Hash    = ByteString
type Type    = ByteString
type Config  = ByteString
type FixedIn = ByteString

-- | Presence condition
--
-- A CNF formula over configs.
type Pc = [[PcLit]]

data PcLit = Pos { litConfig :: !Config }
           | Neg { litConfig :: !Config }

instance Show PcLit where
  show (Pos c) = BC.unpack c
  show (Neg c) = '!' : ' ' : BC.unpack c

noteConfigs :: Note -> [Config]
noteConfigs = nub . map litConfig . concat . notePc

noteLoc0 :: Note -> FilePath
noteLoc0 = head . splitPath . noteLoc

-- YAML fields
f_bugfix, f_hash, f_type, f_config, f_fixin, f_loc :: YamlLight
f_bugfix = YStr "bugfix"
f_hash   = YStr "hash"
f_type   = YStr "type"
f_config = YStr "config"
f_fixin  = YStr "fix-in"
f_loc    = YStr "loc"

-- * Check

checkNotes :: FilePath -> FilePath -> IO ()
checkNotes dbPath prjPath =
  let notesPath = dbPath </> prjPath </> "report" in
  onFilesM notesPath isNoteFile $ \notes_fp -> do
    forM_ notes_fp $ \fp -> do
      putStr $ fp ++ ": "
      note <- readNote notesPath fp
      note `seq` putStrLn "ok"

-- * Parsing

readNote :: FilePath -> FilePath -> IO Note
readNote notesPath fp = do
  f <- B.readFile (notesPath </> fp)
  !y <- Y.parseYamlBytes f
  return $ Note { noteHash   = readHash y
                , noteType   = readType y
                , notePc     = readPc y
                , noteFixed  = readFixed y
                , noteLoc    = readLoc y
                }

getText :: String -> Maybe YamlLight -> ByteString
getText fi y = case y of
  Nothing       -> ppError "getText" fi "not found"
  Just (YStr v) -> v
  Just _        -> ppError "getText" fi "no text"

readHash :: YamlLight -> Hash
readHash y = getText "bugfix/hash" $ do
  c <- Y.lookupYL f_bugfix y
  Y.lookupYL f_hash c

readType :: YamlLight -> Type
readType y = getText "type" $ Y.lookupYL f_type y

readPc :: YamlLight -> Pc
readPc y = parsePc $ getText "config" $ Y.lookupYL f_config y

parsePc :: ByteString -> Pc
parsePc = map parseClause . L.splitOn "&&" . BC.unpack . rmParens
  where parseClause = map (parseLit . trimString) . L.splitOn "||"
        parseLit s | head s == '!' = Neg (BC.pack $ tail s)
                   | otherwise     = Pos (BC.pack s)

readFixed :: YamlLight -> [FixedIn]
readFixed y = parseFixed $ getText "fix-in" $ do
  c <- Y.lookupYL f_bugfix y
  Y.lookupYL f_fixin c

parseFixed :: ByteString -> [FixedIn]
parseFixed = sort . map trimText . BC.split ','

readLoc :: YamlLight -> FilePath
readLoc y = BC.unpack $ getText "loc" $ Y.lookupYL f_loc y

-- * Pretty-printing

class PP a where
  pp :: a -> String

instance PP [Char] where
  pp = id

instance PP ByteString where
  pp = BC.unpack

instance PP Int where
  pp = show

instance PP [FixedIn] where
  pp = BC.unpack . BC.intercalate "+"

instance PP Pc where
  pp = show

instance (PP a, PP b) => PP [(a,b)] where
  pp = intercalate "\n" . map (\(x,y) -> pp x ++ "," ++ pp y)

ppError :: PP x => String -> x -> String -> a
ppError fun x str = error $ fun ++ ": `" ++ pp x ++ "' " ++ str

-- * Utils

headPath :: FilePath -> FilePath
headPath fp = case splitPath fp of
                   []   -> fp
                   p0:_ -> p0

trimString :: String -> String
trimString = filter (\c -> c /= ' ')

trimText :: ByteString -> ByteString
trimText = BC.filter (\c -> c /= ' ')

rmParens :: ByteString -> ByteString
rmParens = BC.filter (\c -> c /= '(' && c /= ')')

